
import os
import sys
import random
from typing import Dict, Any
import hashlib

import json
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk

def _explore_dataset(dataset, hi_tokenizer, en_tokenizer, csv_len = None):
    if csv_len is not None and os.path.exists(csv_len):
        return pd.read_csv(csv_len)

    hi_tokenizer, lang_from = hi_tokenizer
    en_tokenizer, lang_to = en_tokenizer

    df = pd.DataFrame.from_dict(dataset)

    def num_tokens(x):
        hi_len = hi_tokenizer(x[lang_from], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
        en_len = en_tokenizer(x[lang_to], padding=True, truncation=True, return_tensors="pt")['input_ids'].shape[1]
        return hi_len, en_len

    df['hi'], df['en'] = zip(*df.translation.progress_apply(num_tokens))
    df['hi_en'] = df['hi'] + df['en']
    df = df[['hi','en','hi_en']].sort_values('hi_en', ascending=False)
    df['index'] = df.index
    if csv_len:
        df.to_csv(csv_len, index=False)
    return df


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()



class TransformerDataset(Dataset):
    """
        split : 'train', 'validation', 'test'
    """
    def __init__(self, data_params, split, hi_tokenizer, en_tokenizer, max_len = None, subset = None, randomize_subset=True):
        if data_params['dataset_path'][0] == "local":
            self.dataset = load_from_disk(*data_params['dataset_path'][1])[split]
        else:
            self.dataset = load_dataset(*data_params['dataset_path'][1])[split]
        # self.dataset = load_dataset(data_params['dataset_path'])[split]
        self.hi_tokenizer = hi_tokenizer
        self.en_tokenizer = en_tokenizer

        self.lang_from, self.lang_to = data_params.pop('lang', None)

        eda_path = f'eda/{dict_hash(data_params)}'
        seqlen_csv = f"{eda_path}/{split}.csv"

        if not os.path.exists(eda_path):
            os.makedirs(eda_path)
            json.dump(data_params, open(f"{eda_path}/data_params.json", "w"))
        if not os.path.exists(seqlen_csv):
            df = _explore_dataset(self.dataset, (self.hi_tokenizer, self.lang_from), (self.en_tokenizer, self.lang_to), csv_len = seqlen_csv)

        if max_len:
            df = pd.read_csv(seqlen_csv)
            self.dataset = torch.utils.data.Subset(self.dataset, indices = list(df['index'][(df.hi<=max_len//2) & (df.en<=max_len//2)]))

        if subset:
            if randomize_subset:
                subset = random.sample(range(len(self.dataset)), subset)
            else:
                subset = range(subset)
            self.dataset = torch.utils.data.Subset(self.dataset, subset)
        
    def __getitem__(self,index):
        try:
            hi_tokens = self.hi_tokenizer(self.dataset[index]['translation'][self.lang_from], padding=True, truncation=True, return_tensors="pt")
            en_tokens = self.en_tokenizer(self.dataset[index]['translation'][self.lang_to], padding=True, truncation=True, return_tensors="pt")
        except IndexError as er:
            import ipdb; ipdb.set_trace()
            import traceback
            traceback.print_exception(*sys.exc_info())
        return {'hi':hi_tokens, 'en':en_tokens}
    
    def __len__(self):
        return len(self.dataset)
    