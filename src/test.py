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

    if max_len:
        assert token_size_data is not None

        df = pd.read_csv(token_size_data)
        dataset['train'] = torch.utils.data.Subset(dataset['train'], indices = list(df['index'][df.hi_en<=max_len]))

    if subset_len:
        subset = list(range(0, subset_len))
        dataset['train'] = torch.utils.data.Subset(dataset['train'], subset)
        dataset['validation'] = torch.utils.data.Subset(dataset['validation'], subset)

    return dataset


from torch.utils.data import Dataset
class DatasetWithIndex(Dataset):
    def __init__(self, dataset):
        super(DatasetWithIndex, self).__init__()
        self._dataset = dataset

    def __getitem__(self, idx):
        return self._dataset[idx], idx

    def __len__(self):
        return len(self._dataset)


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    hi_tokenizer, en_tokenizer = load_tokenizers(path_hi = "../translation/hindi-tokenizer", path_en = "../translation/eng-tokenizer")
    bs = 5 # batch_size
 
    dataset = prepare_dataset("cfilt/iitb-english-hindi", subset_len = 300, max_len = 300, token_size_data = "train_token_size.csv")
    dataset['train'] = DatasetWithIndex(dataset['train'])

    print("len dataset['train']:",len(dataset['train']))
    print("computing max tokens (dataset['train'])..")
    max_token = -1
    indices = []
    pbar = tqdm(dataset['train'])
    try:
        for b, index in pbar:
            indices.append(index)
            tmp = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1] + en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
            if tmp>max_token:
                max_token = tmp
                pbar.set_description(f"max:{max_token}")
    except  Exception as ex:
        import traceback
        print(''.join(traceback.TracebackException.from_exception(ex).format()))
        import ipdb
        ipdb.set_trace()
    print(f"max token sum: {max_token}")
    print()
    print()

    train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=bs, shuffle=False)

    print("len train_loader:",len(train_loader))
    print("computing max tokens (train_loader)..")
    max_token = -1
    max_index = -1
    pbar = tqdm(train_loader)
    try:
        for b, index in pbar:
            tmp = hi_tokenizer(b['translation']['hi'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1] + en_tokenizer(b['translation']['en'], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
            if tmp>max_token:
                max_token = tmp
                pbar.set_description(f"max:{max_token}")
                max_index = index
    except  Exception as ex:
        import traceback
        print(''.join(traceback.TracebackException.from_exception(ex).format()))
        import ipdb
        ipdb.set_trace()
    print(f"max token sum: {max_token}")
    print()
    print()

 











