import wandb
import numpy as np
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd
import random


GENERATION_ARGS = {
    "output_dir": "./results",
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "logging_dir": './logs',
    "logging_steps": 10,
    "logging_strategy": "steps",
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "disable_tqdm": False,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 2,
    "learning_rate": 5e-5,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "seed": 42,
    "lr_scheduler_type": "linear",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "gradient_accumulation_steps": 1
}


def get_gpt(gpt_path):
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2', truncation=True, padding=True)
    gpt_tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>","bos_token": "<start>", 
                                      "eos_token": "<end>", "unk_token": "<unk>", "ep_token": "<ep>",
                                      "en_token": "<en>", "hp_token": "<hp>", "hn_token": "<hn>"})

    gpt = GPT2LMHeadModel.from_pretrained(gpt_path).cuda()
    gpt.resize_token_embeddings(len(gpt_tokenizer))
    return gpt, gpt_tokenizer


def generate_hard_negative(model, tokenizer, example):
    prompt = example["intent"] + "<sep>" + example["text"] + "<sep>"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to('cuda')

    while True:
        try:
            generated = model.generate(
                input_ids,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
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
    
    return concatenate_datasets(fakes).shuffle()


class GenerationTypedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, distances, hard_num=5, ratios=None):
        self.source = source_dataset
        self.distances = distances
        self.n = len(source_dataset)
        self.ratios = ratios
        if self.ratios is None:
            self.ratios = {
                "ep": 1,
                "en": 1,
                "hp": 1,
                "hn": 1
            }
        self.hard_num = hard_num
    
    def __len__(self):
        return self.n * sum(self.ratios.values())
    
    def __getitem__(self, idx):
        item_idx = idx % self.n
        state = idx // self.n
        anchor = self.source[item_idx]
        to_choose, mode = None, None
        if state < self.ratios["ep"]:
            to_choose = self.source.filter(lambda x: x["intent"] == anchor["intent"])
            mode = "ep"
        elif state < self.ratios["ep"] + self.ratios["en"]:
            to_choose = self.source.filter(lambda x: x["intent"] != anchor["intent"])
            mode = "en"
        elif state < self.ratios["ep"] + self.ratios["en"] + self.ratios["hp"]:
            with_dists = self.source.map(lambda x, idx: {"distance": self.distances[item_idx, idx], **x},
                                         with_indices=True).filter(lambda x: x["intent"] == anchor["intent"])
            best_idxs = np.argpartition(with_dists["distance"], self.hard_num)[:self.hard_num]
            to_choose = with_dists.select(best_idxs)
            mode = "hp"
        else:
            with_dists = self.source.map(lambda x, idx: {"distance": self.distances[item_idx, idx], **x},
                                         with_indices=True).filter(lambda x: x["intent"] != anchor["intent"])
            best_idxs = np.argpartition(with_dists["distance"], -self.hard_num)[:-self.hard_num]
            to_choose = with_dists.select(best_idxs)
            mode = "hn"
        
        other_idx = random.randint(0, len(to_choose))
        other = self.source[other_idx]
        return {
            "source": self.source[item_idx],
            "other": self.source[other_idx],
            "mode": mode
        }
