#!/usr/bin/env python3.7
# coding: utf-8

# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb


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

import transformers
from transformers import AutoTokenizer
from datasets import load_dataset


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
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, pad_token_src = 0, pad_token_tgt = 0, device = 'cpu'):
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
        self.device = device
        self = self.to(self.device)

    def generate_mask(self, src_mask, tgt_mask):
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_mask.unsqueeze(1).unsqueeze(3)
        seq_length = tgt_mask.size(2)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask.to(self.device)
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

        tgt = torch.tensor([[bos_token_id]]*src.shape[0]).to(self.device)
        if mask:
            src_mask, tgt_mask = self.generate_mask(mask['src_mask'], mask['tgt_mask'])
        else:
            src_mask, tgt_mask = self.generate_mask(src!=self.pad_token_src, tgt!=self.pad_token_tgt)
        
        enc_output = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        out_probs = torch.zeros(src.shape[0],0,self.decoder_embedding.num_embeddings).to(self.device)
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
def dataloaders(dataset_path, bs = 2, subset_len = None, max_len = None, token_size_data = None):
    # dataset_path = "cfilt/iitb-english-hindi"

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    dataset = load_dataset(dataset_path)

    if max_len:
        assert token_size_data is not None

        df = pd.read_csv(token_size_data)
        dataset['train'] = torch.utils.data.Subset(dataset['train'], indices = list(df['index'][df.hi_en<=max_len]))

    print("train dataset length:", len(dataset['train']))
    print("validation dataset length:", len(dataset['validation']))
    print("test dataset length:", len(dataset['test']))
    print()

    if subset_len:
        subset = list(range(0, subset_len))
        dataset['train'] = torch.utils.data.Subset(dataset['train'], subset)
        dataset['validation'] = torch.utils.data.Subset(dataset['validation'], subset)

    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=bs, shuffle=True)

    return train_loader, val_loader, test_loader

def train_model(model, hi_tokenizer, en_tokenizer, train_loader, criterion, epochs=10, save_path="./transformer_epoch_{}_batch_{}.pth", save_freq = 2000, log_writer = None):
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train();

    epoch_loss = 0
    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch}: loss : {epoch_loss/len(train_loader):.5f}")
        epoch_loss = 0
        for batch, b in enumerate(pbar):
            if (batch+1)%save_freq==0:
                torch.save(model.state_dict(), save_path.format(epoch,batch))

            try:
                hi_token = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")
                en_token = en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")
                
                hi_input = hi_token['input_ids'].to(device)
                hi_masks = hi_token['attention_mask'].to(device)
                
                en_output = en_token['input_ids'].to(device)
                en_masks = en_token['attention_mask'].to(device)

                optimizer.zero_grad()
                output = model(hi_input, en_output[:, :-1], mask = {'src_mask':hi_masks, 'tgt_mask':en_masks[:, :-1]})
                loss = criterion(output.contiguous().view(-1, len(en_tokenizer)), en_output[:, 1:].contiguous().view(-1))
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            except Exception as error:
                import traceback
                print(''.join(traceback.TracebackException.from_exception(error).format()))
                import ipdb;
                ipdb.set_trace()
                print()

            epoch_loss += loss.item()
            if log_writer:
                log_writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + batch)
                torch.cuda.empty_cache()
                log_writer.add_scalar('memory allocated', torch.cuda.memory_allocated(model.device)/1024/1024, epoch * len(train_loader) + batch)
                log_writer.add_scalar('seq length', hi_input.shape[1], epoch * len(train_loader) + batch)

        torch.save(model.state_dict(), save_path.format(epoch,'N'))


        

# # Evaluate
def evaluate_model(model, hi_tokenizer, en_tokenizer, data_loader, criterion, print_out = False):
    model.eval()
    num_batches = len(data_loader)

    total_loss = 0
    for b in tqdm(data_loader):

        hi_token = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")
        en_token = en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")

        hi_input = hi_token['input_ids'].to(device)
        hi_masks = hi_token['attention_mask'].to(device)

        en_output = en_token['input_ids'].to(device)
        en_masks = en_token['attention_mask'].to(device)

        out_labels, out_probs = model.decode(hi_input, en_tokenizer.bos_token_id, en_tokenizer.eos_token_id, max_dec_length = en_output.shape[1]-1)

        total_loss += criterion(out_probs.contiguous().view(-1, len(en_tokenizer)), en_output[:, 1:].contiguous().view(-1).to(device)).item()
        
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

    epochs = 10
    bs = 20 # batch_size
    save_prefix = 'runs/gpu0_max_tokens_300'
    save_path = save_prefix+"/transformer_epoch_{}_batch_{}.pth"
    save_freq = 5000

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, 
                              d_ff, max_seq_length, dropout, pad_token_src = hi_tokenizer.pad_token_id, 
                              pad_token_tgt = en_tokenizer.pad_token_id, device = device)

    train_loader, val_loader, test_loader = dataloaders("cfilt/iitb-english-hindi", bs = bs, subset_len = None, max_len = 300, token_size_data = "train_token_size.csv")

    criterion = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    writer = SummaryWriter(save_prefix)
    train_model(model, hi_tokenizer, en_tokenizer, train_loader, criterion, epochs=epochs, save_path=save_path, save_freq=save_freq, log_writer = writer)
    writer.close()

    ## Load Model
    model.load_state_dict(torch.load(save_path.format(epochs-1,'N')))

    avg_loss = evaluate_model(model, hi_tokenizer, en_tokenizer, val_loader, criterion, print_out = False)

    print('eval_loss :', avg_loss)



