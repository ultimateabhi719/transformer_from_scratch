#!/usr/bin/env python3.7
# coding: utf-8

import transformers
import bentoml

model= "ultimateabhi/de-en-1"
task = "translation"

bentoml.transformers.save_model(
    task,
    transformers.pipeline(task, model=model),
    metadata=dict(model_name=model),
)