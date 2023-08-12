#!/usr/bin/env python3.7
# coding: utf-8

import os
import argparse
import glob
from natsort import natsorted as nsed

import torch

import pytorch_transformer as pt

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        prog='eval.py',
                        description='evaluate translation model',
                        epilog='evaluate model')
    parser.add_argument('resume_prefix', help='path for saveed model') 
    parser.add_argument('-d','--eval_dataset', help='dataset to evaluate on: train/test/validation') 
    parser.add_argument('-i','--input_text', help='input text (deustche) to translate to english') 
    args = parser.parse_args()


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model_params, train_params = torch.load(os.path.join(args.resume_prefix,"params.pth")) 
    train_params.update({'tokenizers':{'path_hi':"dbmdz/bert-base-german-cased", 'path_en':"bert-base-uncased"}})
    print()
    print("model params:", model_params)

    resume_path = os.path.join(args.resume_prefix,"transformer_epoch_{}_batch_{}.pth".format('*','*'))
    resume_path = nsed(glob.glob(resume_path))[-1]
    print(f"loading model {resume_path} ..")

    hi_tokenizer, en_tokenizer = pt.tokenizer.load_tokenizers(**train_params['tokenizers'])
    model = pt.Transformer(len(hi_tokenizer), len(en_tokenizer), **model_params, 
                        pad_token_src = hi_tokenizer.pad_token_id, 
                        pad_token_tgt = en_tokenizer.pad_token_id)
    model.load_state_dict(torch.load(resume_path, map_location=device)['model_state_dict'])
    model=model.to(device)

    if args.eval_dataset:
        test_dataset = pt.TransformerDataset( ['wmt14', 'de-en'], 
                                            args.eval_dataset, 
                                            (hi_tokenizer,'de'), (en_tokenizer,'en'), 
                                            max_len = 200, 
                                            seqlen_csv = "train_token_size.csv" if args.eval_dataset=='train' else f"train_token_size.{args.eval_dataset}.csv",
                                            subset = range(10))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn = lambda b:pt.collate_tokens(b, hi_tokenizer, en_tokenizer))
    else:
        assert args.input_text is not None
        hi = hi_tokenizer(args.input_text, padding=True, truncation=True, return_tensors="pt") 
        en = en_tokenizer('Google\'s service, offered free of charge, instantly translates words, phrases, and web pages between English and over 100 other languages.', padding=True, truncation=True, return_tensors="pt") 
        test_loader = torch.utils.data.DataLoader([{'hi':hi, 'en':en}], batch_size=1, shuffle=False, collate_fn = lambda b:pt.collate_tokens(b, hi_tokenizer, en_tokenizer))

    criterion = torch.nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_token_id)
    avg_loss = pt.evaluate_model(model, test_loader, criterion, en_tokenizer, print_out = True, device = device)

    print('avg test loss:',avg_loss)