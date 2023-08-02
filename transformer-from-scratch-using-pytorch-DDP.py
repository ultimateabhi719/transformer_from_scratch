#!/usr/bin/env python3.7
# coding: utf-8

# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import numpy as np
import pandas as pd
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from transformers import AutoTokenizer
from datasets import load_dataset

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


# # Transformer Model

# ![transformer.png](attachment:transformer.png)
# ## Build Components

# Position-wise FFN, MHA, Positional Encoding

# ### Multi Headed Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, debug_str = None):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.debug_str = debug_str
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if (self.debug_str == 'cross'):
            print('attn_scores:',attn_scores.shape, mask.shape)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# ### Position-Wise FFN
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# ### Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ## Encoder Layer
# ![encoder.png](attachment:encoder.png)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


# ## Decoder Layer
# ![decoder.png](attachment:decoder.png)
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)#, debug_str="cross")
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


# ## Transformer
# Merging it all together
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, pad_token_src = 0, pad_token_tgt = 0):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.pad_token_src = pad_token_src
        self.pad_token_tgt = pad_token_tgt

    def generate_mask(self, src_mask, tgt_mask):
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3)
        seq_length = tgt_mask.size(2)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(tgt_mask.device)
        return src_mask, tgt_mask

    def decode(self, src, bos_token_id, eos_token_id, mask=None, max_dec_length = 25):
        """
        for inference
        Args:
            src: input to encoder 
            trg: input to decoder
        out:
            out_labels : returns final prediction of sequence
        """

        device_src = src.device

        tgt = torch.tensor([[bos_token_id]]*src.shape[0]).to(device_src)
        if mask:
            src_mask, tgt_mask = self.generate_mask(mask['src_mask'], mask['tgt_mask'])
        else:
            src_mask, tgt_mask = self.generate_mask(src!=self.pad_token_src, tgt!=self.pad_token_tgt)
        
        enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        out_probs = torch.zeros(src.shape[0],0,self.decoder_embedding.num_embeddings).to(device_src)
        out_labels = tgt
        # unfinished_seq = np.array([1]*src.shape[0])
        # i=0; 
        # while (sum(unfinished_seq)>0 & i<max_dec_length):
        for _ in range(max_dec_length):
            dec_output = self.dropout(self.positional_encoding(self.decoder_embedding(out_labels)))
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            output = self.fc(dec_output)

            out_labels = torch.cat((out_labels, output[:,-1:,:].argmax(-1)),dim=1)
            out_probs = torch.cat((out_probs, output[:,-1:,:]),dim=1)

            # unfinished_seq[(out_labels[:,-1] == eos_token_id).cpu().numpy()] = 0
            # i += 1

        assert out_probs.shape[:-1] == out_labels[:,1:].shape
        return out_labels[:,1:], out_probs

    
    def forward(self, src, tgt, mask = None):
        if mask:
            src_mask, tgt_mask = self.generate_mask(mask['src_mask'], mask['tgt_mask'])
        else:
            src_mask, tgt_mask = self.generate_mask(src!=self.pad_token_src, tgt!=self.pad_token_tgt)
                
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

# ## Load Tokenizers from saved
def load_tokenizers(path_hi, path_en):
    # path_hi = "../translation/hindi-tokenizer"
    # path_en = "../translation/eng-tokenizer"
    hi_tokenizer = AutoTokenizer.from_pretrained(path_hi)
    en_tokenizer = AutoTokenizer.from_pretrained(path_en)

    hi_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '<cls>', 'eos_token':'<eos>', 'bos_token' : '<s>'})
    en_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '<cls>', 'eos_token':'<eos>', 'bos_token' : '<s>'})

    from tokenizers.processors import TemplateProcessing
    en_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=en_tokenizer.bos_token + " $A " + en_tokenizer.eos_token,
        special_tokens=[(en_tokenizer.eos_token, en_tokenizer.eos_token_id), (en_tokenizer.bos_token, en_tokenizer.bos_token_id)],
    )
    return hi_tokenizer, en_tokenizer


# ## Create Dataloader
def prepare_dataset(dataset_path, subset_len = None, max_len = None, token_size_data = None):
    # dataset_path = "cfilt/iitb-english-hindi"

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = load_dataset(dataset_path)

    print("input train dataset length:", len(dataset['train']))

    if max_len:
        assert token_size_data is not None

        df = pd.read_csv(token_size_data)
        dataset['train'] = torch.utils.data.Subset(dataset['train'], indices = list(df['index'][df.hi_en<=max_len]))

    if subset_len:
        subset = list(range(0, subset_len))
        dataset['train'] = torch.utils.data.Subset(dataset['train'], subset)
        dataset['validation'] = torch.utils.data.Subset(dataset['validation'], subset)

    print("train dataset length:", len(dataset['train']))
    print("validation dataset length:", len(dataset['validation']))
    print("test dataset length:", len(dataset['test']))

    return dataset

def prepare_dataloader(dataset, rank, world_size, batch_size=32, pin_memory=False, num_workers=0, shuffle = True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=False)
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory, num_workers=num_workers, drop_last=False, shuffle=False, sampler=sampler)
    
    return dataloader

def train_model(rank, world_size, dataset, transformer, hi_tokenizer, en_tokenizer, 
    criterion, bs=None, epochs=10, save_path = "./transformer_epoch_{}_batch_{}.pth", save_freq = 40000, log_path = None, shuffle = True):
    assert bs is not None

    if log_path and rank == 0:
        log_writer = SummaryWriter(log_path)

    
    # if rank==0:
    #     print("len dataset:",len(dataset))
    #     print("computing max tokens (dataset)..")
    #     max_token = -1
    #     pbar = tqdm(dataset)
    #     try:
    #         for b in pbar:
    #             tmp = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1] + en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
    #             if tmp>max_token:
    #                 max_token = tmp
    #                 pbar.set_description(f"max:{max_token}")
    #     except  Exception as ex:
    #         import traceback
    #         print(''.join(traceback.TracebackException.from_exception(ex).format()))
    #         import ipdb
    #         ipdb.set_trace()
    #     print(f"max token sum: {max_token}")
    #     print()
    #     print()

    # setup the process groups
    setup(rank, world_size)
    # prepare the dataloader
    dataloader = prepare_dataloader(dataset, rank, world_size, batch_size=bs, shuffle = shuffle)
    
    # if rank==0:
    #     print("len dataloader:",len(dataloader))
    #     print("computing max tokens (dataloader)..")
    #     max_token = -1
    #     pbar = tqdm(dataloader)
    #     try:
    #         for b in pbar:
    #             tmp = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1] + en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
    #             if tmp>max_token:
    #                 max_token = tmp
    #                 pbar.set_description(f"max:{max_token}")
    #         print("SHAPE", hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape)
    #         print("SHAPE", en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape)
    #     except  Exception as ex:
    #         import traceback
    #         print(''.join(traceback.TracebackException.from_exception(ex).format()))
    #         import ipdb
    #         ipdb.set_trace()
    #     print(f"max token sum: {max_token}")
    #     print()
    #     print()

    # instantiate the model(it's your own model) and move it to the right device
    model = transformer.to(rank)
    
    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train();

    epoch_loss = 0
    for epoch in range(epochs):
        dataloader.sampler.set_epoch(epoch)
        pbar = tqdm(dataloader) if rank == 0 else dataloader
        if rank == 0:
            pbar.set_description(f"Epoch: {epoch}, Loss: {epoch_loss/len(dataloader):.5f} ")
        epoch_loss = 0
        for batch, b in enumerate(pbar):
            if ((batch+1)%save_freq)==0 and rank == 0:
                torch.save(model.module.state_dict(), save_path.format(epoch,batch))

            hi_token = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")
            en_token = en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")
            
            hi_input = hi_token['input_ids']
            hi_masks = hi_token['attention_mask']
            
            en_output = en_token['input_ids']
            en_masks = en_token['attention_mask']
            
            optimizer.zero_grad()
            output = model(hi_input, en_output[:, :-1], mask = {'src_mask':hi_masks, 'tgt_mask':en_masks[:, :-1]})
            loss = criterion(output.contiguous().view(-1, len(en_tokenizer)), en_output[:, 1:].contiguous().view(-1).to(output.device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()

            if log_path and rank == 0:
                log_writer.add_scalar('training loss', loss.item(), epoch * len(dataloader) + batch)
                log_writer.add_scalar('hi_en seq length', hi_input.shape[1]+en_output.shape[1], epoch * len(dataloader) + batch)

        if rank==0:
            torch.save(model.module.state_dict(), save_path.format(epoch,'N'))

    if log_path and rank == 0:
        log_writer.close()

    cleanup()

    

# # Evaluate
def evaluate_model(model, hi_tokenizer, en_tokenizer, data_loader, criterion, print_out = False):
    model.eval()
    num_batches = len(data_loader)

    total_loss = 0
    for b in tqdm(data_loader):

        hi_token = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")
        en_token = en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")

        hi_input = hi_token['input_ids']
        hi_masks = hi_token['attention_mask']

        en_output = en_token['input_ids']
        en_masks = en_token['attention_mask']

        out_labels, out_probs = model.decode(hi_input, en_tokenizer.bos_token_id, en_tokenizer.eos_token_id, max_dec_length = en_output.shape[1]-1)

        total_loss += criterion(out_probs.contiguous().view(-1, len(en_tokenizer)), en_output[:, 1:].contiguous().view(-1)).item()
        
        if print_out:
            print("model:",*list(map(en_tokenizer.decode, out_labels)), sep='\n')
            print("\ntarget:",*list(map(en_tokenizer.decode, en_output)), sep='\n')
            print('------------\n')

    return total_loss/num_batches/en_output.shape[0]



if __name__ == '__main__':
    hi_tokenizer, en_tokenizer = load_tokenizers(path_hi = "../translation/hindi-tokenizer", path_en = "../translation/eng-tokenizer")
    src_vocab_size = len(hi_tokenizer)
    tgt_vocab_size = len(en_tokenizer)
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 1024
    dropout = 0.1

    epochs = 100
    bs = 20 # batch size
    save_prefix = 'runs/ddp_max_tokens_300_subseu_len_50'
    save_path = save_prefix+"/transformer_epoch_{}_batch_{}.pth"
    save_freq = 10000

    resume_path = None


    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, 
                              d_ff, max_seq_length, dropout, pad_token_src = hi_tokenizer.pad_token_id, 
                              pad_token_tgt = en_tokenizer.pad_token_id)
    if resume_path:
        transformer = transformer.load_state_dict(torch.load(resume_path, map_location='cpu'))


    dataset = prepare_dataset("cfilt/iitb-english-hindi", subset_len = 50, max_len = 300, token_size_data = "train_token_size.csv")

    criterion = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)

    world_size = torch.cuda.device_count()
    print(f"using {world_size} GPUs.")
    mp.spawn(
        train_model,
        args=(world_size, dataset['train'], transformer, hi_tokenizer, en_tokenizer, criterion, bs, epochs, save_path, save_freq, save_prefix, False),
        nprocs=world_size
    )

    ## Load Model
    transformer.load_state_dict(torch.load(save_path.format(epochs-1,'N'), map_location='cuda:0'))


    eval_dataloader = prepare_dataloader(dataset['validation'], 0, 1, batch_size=bs, pin_memory=False, num_workers=0)
    avg_loss = evaluate_model(transformer, hi_tokenizer, en_tokenizer, eval_dataloader, criterion, print_out = False)

    print('eval_loss :', avg_loss)












