import wandb
import numpy as np
import torch
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd
import random
from tqdm.auto import tqdm, trange


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
                                      "eos_token": "<end>", "unk_token": "<unk>"})

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


#!g1.1
class GenerationTypedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, distances, hard_num=5, ratios=None):
        self.source = source_dataset
        self.distances = distances
        self.n = len(source_dataset)
        self.intents = self.source.unique("intent")
        self.n_intents = len(self.intents)
        self.ratios = ratios
        if self.ratios is None:
            self.ratios = {
                "ep": 1,
                "en": 1,
                "hp": 1,
                "hn": 1
            }
        self.hard_num = hard_num

        self.idxs_to_choose = {
            "ep": np.zeros((self.n, self.n // self.n_intents), dtype=int),
            "en": np.zeros((self.n, self.n * (self.n_intents - 1) // self.n_intents), dtype=int),
            "hp": np.zeros((self.n, self.hard_num), dtype=int),
            "hn": np.zeros((self.n, self.hard_num), dtype=int),
        }

        self.source = self.source.map(lambda x, idx: {"index": idx, **x}, with_indices=True)
        
        for intent in tqdm(self.intents):
            nice_guys = self.source.filter(lambda x: x["intent"] == intent)
            bad_guys = self.source.filter(lambda x: x["intent"] != intent)
            nice_guys_idx = np.array(nice_guys["index"])
            bad_guys_idx = np.array(bad_guys["index"])

            self.idxs_to_choose["ep"][nice_guys["index"]] = nice_guys["index"]
            self.idxs_to_choose["en"][nice_guys["index"]] = bad_guys["index"]

            inside_dists = self.distances[nice_guys["index"]][:,nice_guys["index"]]
            hard_idxs = np.argpartition(inside_dists, -self.hard_num)[:,-self.hard_num:]
            
            self.idxs_to_choose["hp"][nice_guys["index"]] = nice_guys_idx[hard_idxs]

            outside_dists = self.distances[nice_guys["index"]][:,bad_guys["index"]]
            hard_idxs = np.argpartition(outside_dists, self.hard_num)[:,:self.hard_num]
            self.idxs_to_choose["hn"][nice_guys["index"]] = bad_guys_idx[hard_idxs]
    
    def __len__(self):
        return self.n * sum(self.ratios.values())
    
    def __getitem__(self, idx):
        item_idx = idx % self.n
        state = idx // self.n
        mode = None
        if state < self.ratios["ep"]:
            mode = "ep"
        elif state < self.ratios["ep"] + self.ratios["en"]:
            mode = "en"
        elif state < self.ratios["ep"] + self.ratios["en"] + self.ratios["hp"]:
            mode = "hp"
        else:
            mode = "hn"
        
        other_idx = random.choice(self.idxs_to_choose[mode][item_idx])
        return {
            "source": self.source[item_idx],
            "other": self.source[int(other_idx)],
            "mode": mode
        }
