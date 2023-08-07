
import os
import pandas as pd

from tqdm.auto import tqdm
tqdm.pandas()

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

def _explore_dataset(dataset, hi_tokenizer, en_tokenizer, csv_len = None):
    if csv_len is not None and os.path.exists(csv_len):
        return pd.read_csv(csv_len)

    # dataset = load_dataset(dataset_path)['train']
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

class TransformerDataset(Dataset):
    """
        split : 'train', 'validation', 'test'
    """
    def __init__(self, dataset_path, split, hi_tokenizer, en_tokenizer, max_len = None, seqlen_csv = None, subset = None):
        self.dataset = load_dataset(*dataset_path)[split]
        self.hi_tokenizer, self.lang_from = hi_tokenizer
        self.en_tokenizer, self.lang_to = en_tokenizer

        if max_len:
            assert seqlen_csv is not None

            if not os.path.exists(seqlen_csv):
                df = _explore_dataset(self.dataset, (self.hi_tokenizer, self.lang_from), (self.en_tokenizer, self.lang_to), csv_len = seqlen_csv)

            df = pd.read_csv(seqlen_csv)
            self.dataset = torch.utils.data.Subset(self.dataset, indices = list(df['index'][(df.hi<=max_len//2) & (df.en<=max_len//2)]))

        if subset:
            self.dataset = torch.utils.data.Subset(self.dataset, subset)
        
    def __getitem__(self,index):
        hi_tokens = self.hi_tokenizer(self.dataset[index]['translation'][self.lang_from], padding=True, truncation=True, return_tensors="pt")
        en_tokens = self.en_tokenizer(self.dataset[index]['translation'][self.lang_to], padding=True, truncation=True, return_tensors="pt")

        return {'hi':hi_tokens, 'en':en_tokens}
    
    def __len__(self):
        return len(self.dataset)
    