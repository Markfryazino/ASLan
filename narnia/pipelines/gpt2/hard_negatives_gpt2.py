# Загрузка всякого и создание датасета

import wandb
import numpy as np
import torch
import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict
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

class GenerationNegativeDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, similar, num_similar=2):
        self.source = source_dataset
        self.similar = similar
        self.n = len(source_dataset)
        self.num_similar = num_similar
        self.temp = 20
        self.softmax = torch.nn.Softmax(0)
    
    def __len__(self):
        return self.n * self.num_similar
    
    def __getitem__(self, idx):
        logits = [el[1] for el in self.similar[idx % self.n]]
        other_idxs = [el[0] for el in self.similar[idx % self.n]]
        weights = self.softmax((1 - torch.tensor(logits)) * self.temp)
        bad_id = np.random.choice(other_idxs, p=weights.numpy())
        return {
            "source_text": self.source[idx % self.n]["text"],
            "source_intent": self.source[idx % self.n]["intent"],
            "other_text": self.source[int(bad_id)]["text"],
            "other_intent": self.source[int(bad_id)]["intent"],
        }   

train_dset = GenerationNegativeDataset(raw["seen_train"], train_sim)
val_dset = GenerationNegativeDataset(raw["seen_val"], val_sim)
test_dset = GenerationNegativeDataset(raw["seen_test"], test_sim)


# Собственно обучение GPT-2


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True, padding=True)
tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>", "eos_token": "<end>", "unk_token": "<unk>"})
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

class GPT2GenerationDataset(torch.utils.data.Dataset):
    def __init__(self, source_negative_dataset, tokenizer):
        self.source = source_negative_dataset
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        example = self.source[idx]
        text = "<start>" + example["source_intent"] + "<sep>" + example["source_text"] + "<sep>" + example["other_text"] + "<end>"

        return self.tokenizer(text)

train_final_dset = GPT2GenerationDataset(train_dset, tokenizer)
val_final_dset = GPT2GenerationDataset(val_dset, tokenizer)
test_final_dset = GPT2GenerationDataset(test_dset, tokenizer)

import os
os.environ["WANDB_LOG_MODEL"] = "true"

config = {
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

run = wandb.init(project="aslan",
                 tags=["gpt2", "clinc150", "augmentation", "hard-negatives"],
                 job_type="augmentation-training",
                 config=config)
run.use_artifact("CLINC150:latest")
run.use_artifact("CLINC150-seen-similar:latest")

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
    disable_tqdm=False,
    **config
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_final_dset,
    eval_dataset=val_final_dset,
    data_collator=collator
)

trainer.train()

wandb.finish()
