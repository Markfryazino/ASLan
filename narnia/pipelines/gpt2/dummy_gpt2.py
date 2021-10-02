# This script is a copy of a notebook running in Yandex Datasphere.
# Thus, I am not sure it will work from the box.
# Anyway, you need CLINC150:v4 downloaded.

import wandb
import numpy as np
import torch
import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict
import os
import pandas as pd


# MODEL AND TOKENIZER

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True, padding=True)
tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>", "eos_token": "<end>", "unk_token": "<unk>"})
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))


# DATASET

splits = []
for see in ["seen", "unseen"]:
    for part in ["train", "val", "test"]:
        splits.append(f"{see}_{part}")

raw = load_dataset("csv", data_files={el: f"artifacts/CLINC150:v4/zero_shot_split/{el}.csv" for el in splits})

raw = raw.filter(lambda x: x["intent"] != "oos")
dataset = raw.map(lambda examples: tokenizer("<start>" + examples["intent"] + "<sep>" + examples['text'] + "<end>"), batched=False)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

import os
os.environ["WANDB_LOG_MODEL"] = "true"

config = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 3,
    "learning_rate": 5e-5,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "seed": 42,
    "lr_scheduler_type": "linear",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "gradient_accumulation_steps": 1
}

run = wandb.init(project="aslan",
                 tags=["gpt2", "clinc150", "augmentation"],
                 job_type="augmentation-training",
                 config=config)
run.use_artifact("CLINC150:latest")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir='./logs',
    logging_steps=10,
    logging_strategy="steps",
    report_to="wandb",
    save_strategy="steps",
    save_steps=100,
    save_total_limit=5,
    load_best_model_at_end=True,
    disable_tqdm=True,
    **config
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["seen_train"],
    eval_dataset=dataset["unseen_train"],
    data_collator=collator
)

trainer.train()

wandb.finish()
