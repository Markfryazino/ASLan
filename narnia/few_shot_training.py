import wandb
import numpy as np
import torch
import json
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, \
                         BertTokenizerFast, BertForSequenceClassification, \
                         RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd

from environment import FewShotHandler, TokenizedDataset, UIDataset, UUDataset


ACCURACY = load_metric("accuracy")

COMMON_ARGS = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "max_steps": 150,
    "learning_rate": 2e-5,
    "warmup_steps": 0,
    "weight_decay": 0.01,
    "seed": 42,
    "lr_scheduler_type": "linear",
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "gradient_accumulation_steps": 1,
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


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def model_finetuning(model, tokenizer, fshandler, setup_function, use_artifacts, wandb_tags, 
                     wandb_group, log_model=False, params=None):
    
    if params is None:
        params = {
            "training": {}
        }

    os.environ["WANDB_LOG_MODEL"] = str(log_model).lower()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    model, support_set, test_set = setup_bert(model, tokenizer, fshandler, params)
    
    run = wandb.init(project="aslan", tags=wandb_tags, job_type="training", group=wandb_group)

    wandb.config["support_size"] = len(support_set) / num_labels
    wandb.config["num_labels"] = num_labels

    for artifact in use_artifacts:
        run.use_artifact(artifact)
        run.use_artifact(artifact)

    settings = COMMON_ARGS.copy().update(params["training"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings), 
        train_dataset=support_set,
        eval_dataset=test_set,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    final_metrics = trainer.evaluate(support_set, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(test_set, metric_key_prefix="test"))

    wandb.log(final_metrics)

    run.finish()
    return model


def setup_bert(bert, tokenizer, fshandler, params=None):
    for param in bert.parameters():
        param.requires_grad = True

    num_labels = len(fshandler.intents)

    bert.classifier = torch.nn.Linear(768, num_labels)
    bert.classifier.requires_grad = True
    bert.num_labels = num_labels
    bert.config.num_labels = num_labels

    support_set = TokenizedDataset(fshandler.known, lambda x: x["text"], tokenizer)
    test_set = TokenizedDataset(fshandler.unknown, lambda x: x["text"], tokenizer)

    return bert, support_set, test_set


def setup_entailment_roberta(roberta, tokenizer, fshandler, params):
    support_ui = UIDataset(fshandler.known, fshandler.intents)
    test_ui = UIDataset(fshandler.unknown, fshandler.intents)

    separator = "<sep>"
    if "separator" in params:
        separator = params["separator"]

    support_set = TokenizedDataset(support_ui, lambda x: x["text"] + separator + x["intent"], tokenizer)
    test_set = TokenizedDataset(test_ui, lambda x: x["text"] + separator + x["intent"], tokenizer)

    return roberta, support_set, test_set


def setup_knn_roberta(roberta, tokenizer, fshandler, params):
    support_ui = UUDataset(fshandler.known, fshandler.known)
    test_ui = UUDataset(fshandler.unknown, fshandler.known)

    separator = "<sep>"
    if "separator" in params:
        separator = params["separator"]

    support_set = TokenizedDataset(support_ui, lambda x: x["text_known"] + separator + x["text_unknown"], tokenizer)
    test_set = TokenizedDataset(test_ui, lambda x: x["text_known"] + separator + x["text_unknown"], tokenizer)

    return roberta, support_set, test_set
