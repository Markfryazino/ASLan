# This script is a copy of a notebook running in Yandex Datasphere, it won't work from the box

import numpy as np
from transformers import BertTokenizerFast
from datasets import load_dataset, ClassLabel, load_metric


raw = load_dataset("csv", data_files={"train": "artifacts/CLINC150:v1/train.csv",
                                      "val": "artifacts/CLINC150:v1/val.csv",
                                      "test": "artifacts/CLINC150:v1/test.csv", })

no_oos = raw.filter(lambda x: x["intent"] != "oos")

tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
tokenized_dataset = no_oos.map(lambda examples: tokenizer(examples['text'], padding="max_length", truncation=True), batched=True)

unique_intents = list(tokenized_dataset.unique("intent")["train"])
classlabel = ClassLabel(names=unique_intents)

def encode_label(x):
    x["label"] = classlabel.str2int(x["intent"])
    return x

final_dataset = tokenized_dataset.map(encode_label)

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


%env WANDB_LOG_MODEL=true

import wandb
from transformers import BertForSequenceClassification, Trainer, TrainingArguments


config = {
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 5,
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
                 tags=["bert", "clinc150", "full-data"],
                 job_type="training",
                 group="bert-on-clinc-baseline",
                 config=config)
run.use_artifact("CLINC150:v1")

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
    metric_for_best_model="accuracy",
    greater_is_better=True,
    disable_tqdm=True,
    **config
)

model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(unique_intents))

trainer = Trainer(
    model=model,
    args=training_args, 
    train_dataset=final_dataset["train"],
    eval_dataset=final_dataset["val"],
    compute_metrics=compute_metrics
)
trainer.train()

final_metrics = trainer.evaluate(final_dataset["train"], metric_key_prefix="train")
final_metrics.update(trainer.evaluate(final_dataset["val"], metric_key_prefix="val"))
final_metrics.update(trainer.evaluate(final_dataset["test"], metric_key_prefix="test"))

wandb.log(final_metrics)

run.finish()