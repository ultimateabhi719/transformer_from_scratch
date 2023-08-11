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

    train_params = {
        # 'dataset_path' : "cfilt/iitb-english-hindi",
        'dataset_path' : ['wmt14', 'de-en'],
        'lang_from' : 'de',
        'lang_to' : 'en',
        'max_len' : 30,
        'subset' : range(340000),
        'subset_eval' : range(40),

        'learning_rate' : 1e-4,
        'epochs' : 40,
        'batch_size' : 84,
        'save_freq' : 10000, #batches
        'logwt_freq' : 400, #batches # set to 0 to stop weight logging

        'save_prefix' : 'runs/de_en_maxlen30_subset340k_log/',
        'save_path' : "transformer_epoch_{}_batch_{}.pth",
        'resume_path' : 'runs/de_en_subset62k_log4/transformer_epoch_14_batch_N.pth',
        'fresh_init' : False,

        'batch_size_val' : 20,
        'batch_size_test' : 20
    }

    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    os.makedirs(train_params['save_prefix'], exist_ok=True)
    torch.save([model_params,train_params],train_params['save_prefix']+'params.pth')

    # params = torch.load(train_params['save_prefix']+"params.pth")

    pytorch_transformer.main(model_params, train_params, device)