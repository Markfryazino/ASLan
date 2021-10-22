#!g1.1

import wandb
import numpy as np
import torch
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict
import os
import pandas as pd


def cut_indices(dset, support_size, state):
    np.random.seed(state)
    return list(map(int, list(pd.Series(dset["label"]).rename("label").reset_index(drop=False).groupby("label") \
                ["index"].apply(lambda x: np.random.choice(x, size=support_size)).explode().sample(frac=1))))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


os.environ["WANDB_LOG_MODEL"] = "false"

dataset = DatasetDict.load_from_disk("datasets/data")

with open("datasets/mapping.json") as f:
    mapping = json.load(f)
    
with open("datasets/train_idxs.json") as f:
    train_idxs = json.load(f)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
metric = load_metric("accuracy")

config = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "max_steps": 200,
    "learning_rate": 2e-5,
    "warmup_steps": 0,
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
    "eval_steps": 10,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 10,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "greater_is_better": True,
    "disable_tqdm": True
}

support_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 70, 100]
random_states = [1, 2, 3, 4, 5]

def pipeline(support_size, state):
    model = BertForSequenceClassification.from_pretrained("artifacts/model-m1aqtxai:v0", num_labels=len(mapping["train"]["int2str"]))
    model.num_labels = len(mapping["test"]["int2str"])
    model.config.num_labels = len(mapping["test"]["int2str"])
    model.classifier = torch.nn.Linear(model.config.hidden_size, model.config.num_labels)

    # DtestStrain = dataset["DtestStrain"].select(train_idxs[str(support_size)]["DtestStrain"])
    DtestStrain = dataset["DtestStrain"].select(cut_indices(dataset["DtestStrain"], support_size, state))
    
    run = wandb.init(project="aslan",
                     tags=["bert", "clinc150", "finetuning-on-Dtest"],
                     job_type="training",
                     group="short-finetuning-on-Dtest-no-saving",
                     config=config)
    wandb.config["support_size"] = support_size
    wandb.config["random_state"] = state
    run.use_artifact("CLINC150-tokenized:latest")
    run.use_artifact("model-m1aqtxai:latest")

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**config, **common_args), 
        train_dataset=DtestStrain,
        eval_dataset=dataset["DtestSval"],
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # final_metrics = trainer.evaluate(dataset["DtestStrain"], metric_key_prefix="train")
    final_metrics = trainer.evaluate(DtestStrain, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(dataset["DtestSval"], metric_key_prefix="val"))
    final_metrics.update(trainer.evaluate(dataset["DtestStest"], metric_key_prefix="test"))

    wandb.log(final_metrics)

    run.finish()
    
for size in support_sizes:
    for state in random_states:
        pipeline(size, state)
