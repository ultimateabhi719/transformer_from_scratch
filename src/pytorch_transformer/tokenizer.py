
import torch 

from transformers import AutoTokenizer
from datasets import load_dataset

def _collate_batch(batch, lang):
    batch_tensor = []
    for b in batch:
        batch_tensor.append(b['translation'][lang])
    return batch_tensor

def _train_tokenizer(dataset_path, lang_from = 'hi', lang_to = 'en', batch_size=1000, shuffle = True, hi_vocab_size = 75000, en_vocab_size = 50000, save_paths = None):
    """
    Function 
        dataset_path = "cfilt/iitb-english-hindi"
        batch_size = 1000
        shuffle = False
        hi_vocab_size = 75000
        en_vocab_size = 50000
        save_paths = {'hi':"trained_tokenizers/hi-tokenizer", 'en':"trained_tokenizers/en-tokenizer"}
    """

    dataset = load_dataset(dataset_path)

    dataloader_hi = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, collate_fn=lambda b:_collate_batch(b,lang_from), shuffle=shuffle)
    dataloader_en = torch.utils.data.DataLoader(dataset["train"], batch_size=batch_size, collate_fn=lambda b:_collate_batch(b,lang_to), shuffle=shuffle)

    old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

    hi_tokenizer = old_tokenizer.train_new_from_iterator(dataloader_hi, hi_vocab_size)
    en_tokenizer = old_tokenizer.train_new_from_iterator(dataloader_en, en_vocab_size)

    if save_paths:
        hi_tokenizer.save_pretrained(save_paths['hi'])
        en_tokenizer.save_pretrained(save_paths['en'])

    return hi_tokenizer, en_tokenizer


def load_tokenizers(path_hi=None, path_en=None):
    """
    Load Tokenizers from saved
        path_hi = "trained_tokenizers/hi-tokenizer"
        path_en = "trained_tokenizers/en-tokenizer"
    """
    hi_tokenizer = AutoTokenizer.from_pretrained(path_hi)
    en_tokenizer = AutoTokenizer.from_pretrained(path_en)

    hi_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'eos_token':'[EOS]', 'bos_token' : '[BOS]'})
    en_tokenizer.add_special_tokens({'pad_token': '[PAD]', 'cls_token': '[CLS]', 'eos_token':'[EOS]', 'bos_token' : '[BOS]'})

    from tokenizers.processors import TemplateProcessing
    en_tokenizer._tokenizer.post_processor = TemplateProcessing(
        single=en_tokenizer.bos_token + " $A " + en_tokenizer.eos_token,
        special_tokens=[(en_tokenizer.eos_token, en_tokenizer.eos_token_id), (en_tokenizer.bos_token, en_tokenizer.bos_token_id)],
    )
    return hi_tokenizer, en_tokenizer
