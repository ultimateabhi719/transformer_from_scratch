#!/usr/bin/env python3.7
# coding: utf-8

import os
import sys
# sys.path.append("src")
import glob
from natsort import natsorted

import torch
import torch.nn as nn
import bentoml

import pytorch_transformer as pt


class BentoTransformer(nn.Module):
    def __init__(self, resume_dir):
        super(BentoTransformer, self).__init__()

        # instantiate model
        model_params, data_params, train_params = torch.load(os.path.join(resume_dir,'params.pth'))
        self.hi_tokenizer, self.en_tokenizer = pt.tokenizer.load_tokenizers(**data_params['tokenizers'])
        self.model = pt.Transformer(len(self.hi_tokenizer), len(self.en_tokenizer), **model_params, 
                            pad_token_src = self.hi_tokenizer.pad_token_id, 
                            pad_token_tgt = self.en_tokenizer.pad_token_id)

        # load latest model in resume_dir or if the whole path is specified 
        if resume_dir[-4:] == ".pth":
            resume_file = resume_dir
        else:
            resume_file = os.path.join(resume_dir,train_params['save_format'].format('*','*'))

        resume_file = natsorted(glob.glob(resume_file))[-1]
        print("loading model:",resume_file)
        resume_dict = torch.load(resume_file,map_location='cpu')
        self.model.load_state_dict(resume_dict['model_state_dict'])

    def predict(self, input_text, max_length = 28):
        hi_token = self.hi_tokenizer(input_text, padding=True, truncation=True, return_tensors="pt") 
        out_labels, _ = self.model.decode(hi_token['input_ids'], self.en_tokenizer.bos_token_id, self.en_tokenizer.eos_token_id, max_dec_length = max_length)
        return list(map(self.en_tokenizer.decode, out_labels))[0]

    def forward(self, src, tgt, mask = None):
        return self.model(src, tgt, mask = mask)

if __name__=="__main__":

    resume_dir="runs/de_en_maxlen28_log3"
    bentoTransformer = BentoTransformer(resume_dir=resume_dir)

    bentoml.pytorch.save_model(
        "translate_de_en_old",   # model name in the local model store
        bentoTransformer,  # model instance being saved
        labels={    # user-defined labels for managing models in Yatai
            "owner": "ultimateabhi",
            "stage": "dev",
        },
        metadata={  # user-defined additional metadata
            "dataset": "wmt14, de-en",
        },
        custom_objects={    # save additional user-defined python objects
            "hi_tokenizer": bentoTransformer.hi_tokenizer,
            'en_tokenizer': bentoTransformer.en_tokenizer
        },
        signatures={   # model signatures for runner inference
            "predict": {
             "batchable": False,
            }
        }
    )
