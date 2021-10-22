import wandb
import numpy as np
import torch
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, RobertaTokenizerFast, \
                         RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd


wandb.login()

api = wandb.Api()
artifact = api.artifact('broccoliman/aslan/CLINC150-seen-similar:latest')
artifact.download()

artifact = api.artifact('broccoliman/aslan/CLINC150:latest')
artifact.download()

with open("artifacts/CLINC150-seen-similar:v1/intents.json") as f:
    intents_sim = json.load(f)
    
with open("artifacts/CLINC150-seen-similar:v1/utterances.json") as f:
    train_sim = json.load(f)
    
with open("artifacts/CLINC150-seen-similar:v1/utterances_val.json") as f:
    val_sim = json.load(f)
    
with open("artifacts/CLINC150-seen-similar:v1/utterances_test.json") as f:
    test_sim = json.load(f)

splits = []
for part in ["train", "val", "test"]:
    splits.append(f"seen_{part}")

raw = load_dataset("csv", data_files={el: f"artifacts/CLINC150:v4/zero_shot_split/{el}.csv" for el in splits})
raw = raw.filter(lambda x: x["intent"] != "oos")

import random

class PositiveNegativeDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, similar):
        self.source = source_dataset
        self.similar = similar
        self.n = len(source_dataset)
        self.temp = 20
        self.softmax = torch.nn.Softmax(0)

        unique_intents = source_dataset.unique("intent")
        self.intent_texts = {}
        for intent in unique_intents:
            self.intent_texts[intent] = source_dataset.select(np.where(np.array(source_dataset["intent"]) == intent)[0])["text"]
    
    def __len__(self):
        return self.n * 2
    
    def __getitem__(self, idx):
        if idx < self.n:
            return {
                "source_text": self.source[idx]["text"],
                "source_intent": self.source[idx]["intent"],
                "other_text": random.choice(self.intent_texts[self.source[idx]["intent"]]),
                "other_intent": self.source[idx]["intent"],
                "label": 1
            }
        else:
            logits = [el[1] for el in self.similar[idx % self.n]]
            other_idxs = [el[0] for el in self.similar[idx % self.n]]
            weights = self.softmax((1 - torch.tensor(logits)) * self.temp)
            bad_id = np.random.choice(other_idxs, p=weights.numpy())
            return {
                "source_text": self.source[idx % self.n]["text"],
                "source_intent": self.source[idx % self.n]["intent"],
                "other_text": self.source[int(bad_id)]["text"],
                "other_intent": self.source[int(bad_id)]["intent"],
                "label": 0
            }   

train_dset = PositiveNegativeDataset(raw["seen_train"], train_sim)
val_dset = PositiveNegativeDataset(raw["seen_val"], val_sim)
test_dset = PositiveNegativeDataset(raw["seen_test"], test_sim)

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, source_negative_dataset, tokenizer):
        self.source = source_negative_dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        example = self.source[idx]
        text = example["source_text"] + "<sep>" + example["other_text"]

        return {**self.tokenizer(text), "label": example["label"]}

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>", "eos_token": "<end>", "unk_token": "<unk>"})
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

train_final_dset = TokenizedDataset(train_dset, tokenizer)
val_final_dset = TokenizedDataset(val_dset, tokenizer)
test_final_dset = TokenizedDataset(test_dset, tokenizer)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

os.environ["WANDB_LOG_MODEL"] = "true"

metric = load_metric("accuracy")

config = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "max_steps": 2000,
    "learning_rate": 5e-5,
    "warmup_steps": 300,
    "weight_decay": 0.01,
    "seed": 42,
    "lr_scheduler_type": "linear",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "gradient_accumulation_steps": 1
}

common_args = {
    "output_dir": "./results",
    "evaluation_strategy": "steps",
    "eval_steps": 100,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 100,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "disable_tqdm": False
}

run = wandb.init(project="aslan",
                 tags=["bert", "clinc150", "roberta-fewshot"],
                 job_type="training",
                 group="few-shot-roberta",
                 config=config)
run.use_artifact("CLINC150:latest")
run.use_artifact("CLINC150-seen-similar:latest")

trainer = Trainer(
    model=model,
    args=TrainingArguments(**config, **common_args), 
    train_dataset=train_final_dset,
    eval_dataset=val_final_dset,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()

final_metrics = trainer.evaluate(train_final_dset, metric_key_prefix="train")
final_metrics.update(trainer.evaluate(val_final_dset, metric_key_prefix="val"))
final_metrics.update(trainer.evaluate(test_final_dset, metric_key_prefix="test"))

wandb.log(final_metrics)

run.finish()
