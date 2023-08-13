#!/usr/bin/env python3.7
# coding: utf-8

import sys
import torch
import torch.nn as nn

import bentoml

translation_model = bentoml.models.get("translate_de_en_old:latest")

translation_runner = translation_model.to_runner()
svc = bentoml.Service(
    name="translate_de_en", runners=[translation_runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def translate(text: str) -> str:
    return translation_runner.predict.run(text)
