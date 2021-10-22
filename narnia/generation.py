import wandb
import numpy as np
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd


def get_gpt(gpt_path):
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', truncation=True, padding=True)
    gpt_tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>","bos_token": "<start>", 
                                      "eos_token": "<end>", "unk_token": "<unk>"})
    gpt_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    gpt = GPT2LMHeadModel.from_pretrained("artifacts/hard-negatives-gpt2:v0").cuda()
    gpt.resize_token_embeddings(len(tokenizer))
    return gpt, gpt_tokenizer


def generate_hard_negative(model, tokenizer, example):
    prompt = example["intent"] + "<sep>" + example["text"] + "<sep>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    while True:
        try:
            generated = model.generate(
                input_ids,
                pad_token_id=gpt_tokenizer.pad_token_id,
                eos_token_id=gpt_tokenizer.eos_token_id,
                max_length=50,
                do_sample=True,
                temperature=.8
            )

            result = tokenizer.decode(generated[0], skip_special_tokens=False)
            example["generated"] = result.split("<sep>")[-1].replace("<end>", "").strip()
            return example
        except:
            print("Something wrong...")


def synthesize_hard_negatives(dataset, model, tokenizer, ratio=1):
    fakes = []
    for i in range(ratio):
        fakes.append(dataset.map(lambda x: generate_hard_negative(model, tokenizer, x)))
    
    return concatenate_datasets([dataset] + [fakes]).shuffle()
