#!/usr/bin/env python3.7
# coding: utf-8

import sys
import torch
import torch.nn as nn

import bentoml

translation_model = bentoml.models.get("translate_hi_en:latest")

translation_runner = translation_model.to_runner()
svc = bentoml.Service(
    name="translate_hi_en", runners=[translation_runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
def translate(text: str) -> str:
    return translation_runner.predict.run(text)


# bentoml serve -p 3000 --api-workers 1 --working-dir scripts/bentoml service:svc