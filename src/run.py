#!/usr/bin/env python3.7
# coding: utf-8

import os
import torch

import pytorch_transformer

if __name__ == '__main__':
    model_params = {
        'd_model' : 512,
        'num_heads' : 8,
        'num_layers' : 6,
        'd_ff' : 2048,
        'max_seq_length' : 1024,
        'dropout' : 0.1
        }

    data_params = { 'dataset_path' : ['local',["data/cvit-pib.hf"]],
                'tokenizers':{'path_hi':'monsoon-nlp/hindi-bert', 'path_en':'bert-base-uncased'},
                'lang' : ('hi','en') #(from-language, to-language)
              }

    train_params = {
        'save_format' : "transformer_epoch_{}_batch_{}.pth",
        'max_len' : 76,
        'subset' : None,
        'subset_eval' : None,

        'learning_rate' : 1e-4,
        'epochs' : 40,
        'batch_size' : 216,
        'save_freq' : 10000, #batches
        'logwt_freq' : 175, #batches # set to 0 to stop weight logging

        'save_prefix' : 'runs/hi_en_maxlen76_cvit_log',
        'resume_dir' : None,
        'fresh_init' : True,

        'batch_size_val' : 20,
        'batch_size_test' : 20
    }

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    os.makedirs(train_params['save_prefix'], exist_ok=True)
    torch.save([model_params, data_params, train_params],os.path.join(train_params['save_prefix'],'params.pth'))

    # params = torch.load(train_params['save_prefix']+"params.pth")

    pytorch_transformer.main(model_params, data_params, train_params, device)