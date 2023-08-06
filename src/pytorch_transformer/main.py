#!/usr/bin/env python3.7
# coding: utf-8

from tqdm.auto import tqdm
import itertools

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim

from .tokenizer import load_tokenizers
from .transformer import Transformer
from .dataset import TransformerDataset

def save_model(x0, model, optimizer, save_path):
    torch.save({
            'x0': x0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)

def train_one_epoch(model, device, train_loader, criterion, optimizer, epoch=0, save_path="transformer_epoch_{}_batch_{}.pth", save_freq = 2000, log_writer = None, x0 = 0):
    model.train();

    pbar = tqdm(train_loader)
    epoch_loss = 0
    running_loss = 0
    for batch_idx, data in enumerate(pbar):
        hi_input = data['hi']['input_ids'].to(device)
        hi_masks = data['hi']['attention_mask'].to(device)
        
        en_output = data['en']['input_ids'].to(device)
        en_masks = data['en']['attention_mask'].to(device)

        optimizer.zero_grad()
        output = model(hi_input, en_output[:, :-1], mask = {'src_mask':hi_masks, 'tgt_mask':en_masks[:, :-1]})
        loss = criterion(output.contiguous().view(-1, output.shape[-1]), en_output[:, 1:].contiguous().view(-1))
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    
        running_loss += loss.item()

        if (batch_idx+1)%save_freq==0:
            save_model(x0 + epoch * len(train_loader) + batch_idx, model, optimizer, save_path.format(epoch,batch_idx))
            pbar.set_description(f"Epoch {epoch}: loss {running_loss/save_freq:.3f}")
            epoch_loss += running_loss
            running_loss = 0

        if log_writer:
            log_writer.add_scalar('training loss', loss.item(), x0 + epoch * len(train_loader) + batch_idx)
            log_writer.add_scalar('hi_en seq length', hi_input.shape[1]+en_output.shape[1], x0 + epoch * len(train_loader) + batch_idx)

    if len(train_loader)%save_freq!=0:
        epoch_loss += running_loss

    save_model(x0 + epoch * len(train_loader) + batch_idx, model, optimizer, save_path.format(epoch,'N'))
    # torch.save(model.state_dict(), save_path.format(epoch,'N'))

    return epoch_loss/len(train_loader)

def train_model(model, device, lr, train_loader, eval_loader, criterion, ent, epochs=10, save_path="transformer_epoch_{}_batch_{}.pth", save_freq = 2000, log_writer = None, 
    resume_dict = {'x0':0, 'optimizer_state_dict':None}):

    x0 = resume_dict['x0']
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    if resume_dict['optimizer_state_dict']:
        optimizer.load_state_dict(resume_dict['optimizer_state_dict'])

    model.train();

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, device, train_loader, criterion, optimizer, epoch=epoch, save_path=save_path, save_freq = save_freq, log_writer = log_writer, x0 = x0)
        val_loss = evaluate_model(model, eval_loader, criterion, ent, print_out = False, device = device)
        if log_writer:
            log_writer.add_scalar('val loss', val_loss, x0 + epoch * len(train_loader))
        print(f"Epoch {epoch}: \n\t\ttrain_loss: {train_loss:.3f}, val_loss: {val_loss:.3f}")


# # Evaluate
@torch.no_grad()
def evaluate_model(model, eval_dataloader, criterion, ent, print_out = False, device = 'cuda:0'):
    model.eval()

    total_loss = 0
    for b in tqdm(eval_dataloader):

        hi_input = b['hi']['input_ids'].to(device)
        hi_masks = b['hi']['attention_mask'].to(device)

        en_output = b['en']['input_ids'].to(device)
        en_masks = b['en']['attention_mask'].to(device)

        out_labels, out_probs = model.decode(hi_input, ent.bos_token_id, ent.eos_token_id, max_dec_length = en_output.shape[1]-1)

        total_loss += criterion(out_probs.contiguous().view(-1, out_probs.shape[-1]), en_output[:, 1:].contiguous().view(-1).to(device)).item()
        
        if print_out:
            print("model:",*list(map(ent.decode, out_labels)), sep='\n')
            print("\ntarget:",*list(map(ent.decode, en_output)), sep='\n')
            print('------------\n')

    return total_loss/len(eval_dataloader)


def collate_tokens(batch, hit, ent):
    out = {}
    out['hi'] = {}
    out['en'] = {}

    def _collate(batch, key1, key2, fillvalue = 0):
        return torch.tensor(list(zip(*itertools.zip_longest(*[b[key1][key2][0] for b in batch], fillvalue=fillvalue))))
    
    out['hi']['input_ids'] = _collate(batch, 'hi', 'input_ids', fillvalue = hit.pad_token_id)
    out['hi']['attention_mask'] = _collate(batch, 'hi', 'attention_mask')

    out['en']['input_ids'] =  _collate(batch, 'en', 'input_ids', fillvalue = ent.pad_token_id)
    out['en']['attention_mask'] =  _collate(batch, 'en', 'attention_mask')

    return out

def main(model_params, train_params, device):
    save_path = train_params['save_prefix']+train_params['save_path']
    lang_from = train_params['lang_from']
    lang_to = train_params['lang_to']

    hi_tokenizer, en_tokenizer = load_tokenizers()
    model = Transformer(len(hi_tokenizer), len(en_tokenizer), **model_params, 
                        pad_token_src = hi_tokenizer.pad_token_id, 
                        pad_token_tgt = en_tokenizer.pad_token_id).to(device)
    if train_params['resume_path']:
        resume_dict = torch.load(train_params['resume_path'])
        model.load_state_dict(resume_dict['model_state_dict'])

    criterion = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)


    train_dataset = TransformerDataset( train_params['dataset_path'], 
                                        'train', 
                                        (hi_tokenizer,lang_from), (en_tokenizer,lang_to), 
                                        max_len = train_params['max_len'], 
                                        seqlen_csv = "train_token_size.csv", 
                                        subset = train_params['subset'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, collate_fn = lambda b:collate_tokens(b, hi_tokenizer, en_tokenizer))

    val_dataset = TransformerDataset( train_params['dataset_path'], 
                                        'validation', 
                                        (hi_tokenizer,lang_from), (en_tokenizer,lang_to), 
                                        max_len = None, 
                                        seqlen_csv = None, 
                                        subset = train_params['subset_eval'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn = lambda b:collate_tokens(b, hi_tokenizer, en_tokenizer))

    writer = SummaryWriter(train_params['save_prefix'])
    train_model(model, device, train_params['learning_rate'], train_loader, val_loader, criterion, en_tokenizer,
        epochs=train_params['epochs'], 
        save_path=save_path, 
        save_freq=train_params['save_freq'], 
        log_writer = writer,
        resume_dict = resume_dict if train_params['resume_path'] else {'x0':0, 'optimizer_state_dict':None})
    writer.close()

    # ## Load Model
    # model.load_state_dict(torch.load(save_path.format(epochs-1,'N')))

    test_dataset = TransformerDataset( train_params['dataset_path'], 
                                        'test', 
                                        (hi_tokenizer,lang_from), (en_tokenizer,lang_to), 
                                        max_len = None, 
                                        seqlen_csv = None, 
                                        subset = None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn = lambda b:collate_tokens(b, hi_tokenizer, en_tokenizer))
    avg_loss = evaluate_model(model, test_loader, criterion, en_tokenizer, print_out = False, device = device)
    print('test_loss :', avg_loss)


if __name__ == '__main__':
    main()