import wandb
import numpy as np
import torch
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, RobertaTokenizerFast, \
                         RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
splits = []
for part in ["train", "val", "test"]:
    splits.append(f"seen_{part}")

raw = load_dataset("csv", data_files={el: f"artifacts/CLINC150:v4/zero_shot_split/{el}.csv" for el in splits})
raw = raw.filter(lambda x: x["intent"] != "oos")

dataset = raw.map(lambda examples: tokenizer(examples["text"]), batched=True)
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

with open("similar_data/intents.json") as f:
    intents_sim = json.load(f)
    
with open("similar_data/utterances.json") as f:
    train_sim = json.load(f)
    
with open("similar_data/utterances_val.json") as f:
    val_sim = json.load(f)
    
with open("similar_data/utterances_test.json") as f:
    test_sim = json.load(f)

class EntailmentDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, intent_similar, utterance_similar, tokenizer, num_sim_i=2, num_sim_u=2, debug=False):
        self.source = source_dataset
        self.intents = intent_similar
        self.utterances = utterance_similar
        self.n = len(source_dataset)
        self.num_sim_i = 2
        self.num_sim_u = 2
        self.tokenizer = tokenizer
        self.i_temp = 10
        self.u_temp = 20
        self.softmax = torch.nn.Softmax(0)
        self.debug = debug
        
    def merge_texts(self, text, intent):
        prompt = text + "<sep>" + intent
        return self.tokenizer(prompt)

    def create_example(self, text, intent, label):
        tokenized = self.merge_texts(text, intent)
        res = {"input_ids": torch.tensor(tokenized["input_ids"]), "attention_mask": torch.tensor(tokenized["attention_mask"]),
               "label": torch.tensor(label)}
        if self.debug:
            res["debug"] = text + " <sep> " + intent
        return res
    
    def __len__(self):
        return self.n * (1 + self.num_sim_i + self.num_sim_u)
    
    def __getitem__(self, idx):
        typ = idx // self.n
        if typ == 0:
            return self.create_example(self.source[idx]["text"], self.source[idx]["intent"], 1)
        elif typ < 1 + self.num_sim_i:
            logits = [el[1] for el in self.intents[self.source[idx % self.n]["intent"]]]
            names = [el[0] for el in self.intents[self.source[idx % self.n]["intent"]]]
            weights = self.softmax((1 - torch.tensor(logits)) * self.i_temp)
            intent = np.random.choice(names, p=weights.numpy())
            return self.create_example(self.source[idx % self.n]["text"], intent, 0)
        else:
            logits = [el[1] for el in self.utterances[idx % self.n]]
            other_idxs = [el[0] for el in self.utterances[idx % self.n]]
            weights = self.softmax((1 - torch.tensor(logits)) * self.u_temp)
            bad_id = np.random.choice(other_idxs, p=weights.numpy())
            return self.create_example(self.source[int(bad_id)]["text"], self.source[idx % self.n]["intent"], 0)     

train_dset = EntailmentDataset(dataset["seen_train"], intents_sim, train_sim, tokenizer, debug=False)
val_dset = EntailmentDataset(dataset["seen_val"], intents_sim, val_sim, tokenizer, debug=False)
test_dset = EntailmentDataset(dataset["seen_test"], intents_sim, test_sim, tokenizer, debug=False)

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
    "disable_tqdm": True
}

run = wandb.init(project="aslan",
                 tags=["bert", "clinc150", "roberta-zeroshot"],
                 job_type="training",
                 group="zero-shot-roberta",
                 config=config)
run.use_artifact("CLINC150:latest")
run.use_artifact("CLINC150-seen-similar:latest")

trainer = Trainer(
    model=model,
    args=TrainingArguments(**config, **common_args), 
    train_dataset=train_dset,
    eval_dataset=val_dset,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()

final_metrics = trainer.evaluate(train_dset, metric_key_prefix="train")
final_metrics.update(trainer.evaluate(val_dset, metric_key_prefix="val"))
final_metrics.update(trainer.evaluate(test_dset, metric_key_prefix="test"))

wandb.log(final_metrics)

run.finish()
