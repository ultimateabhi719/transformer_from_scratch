#!/usr/bin/env python3.7
# coding: utf-8

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
        'max_len' : 300,
        'subset' : range(50),
        'subset_eval' : range(8),

        'learning_rate' : 1e-4,
        'epochs' : 100,
        'batch_size' : 20,
        'save_freq' : 5000,

        'save_prefix' : 'runs/test4/',
        'save_path' : "transformer_epoch_{}_batch_{}.pth",
        'resume_path' : None#'runs/test2/transformer_epoch_2_batch_N.pth'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    pytorch_transformer.main(model_params, train_params, device)