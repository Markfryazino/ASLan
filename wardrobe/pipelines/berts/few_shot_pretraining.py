#!g1.1

import wandb
import numpy as np
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict
import os


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


os.environ["WANDB_LOG_MODEL"] = "true"

dataset = DatasetDict.load_from_disk("datasets/data")

with open("datasets/mapping.json") as f:
    mapping = json.load(f)
    
with open("datasets/train_idxs.json") as f:
    train_idxs = json.load(f)
    
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(mapping["train"]["int2str"]))
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
metric = load_metric("accuracy")

config = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "max_steps": 3000,
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
                 tags=["bert", "clinc150", "pretraining-on-Dtrain"],
                 job_type="training",
                 group="bert-on-clinc-baseline",
                 config=config)
run.use_artifact("CLINC150-tokenized:latest")

trainer = Trainer(
    model=model,
    args=TrainingArguments(**config, **common_args), 
    train_dataset=dataset["DtrainStrain"],
    eval_dataset=dataset["DtrainSval"],
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()

final_metrics = trainer.evaluate(dataset["DtrainStrain"], metric_key_prefix="train")
final_metrics.update(trainer.evaluate(dataset["DtrainSval"], metric_key_prefix="val"))
final_metrics.update(trainer.evaluate(dataset["DtrainStest"], metric_key_prefix="test"))

wandb.log(final_metrics)

run.finish()