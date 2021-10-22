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
PRECISION = load_metric("precision")
RECALL = load_metric("recall")
F1 = load_metric("f1")

COMMON_ARGS = {
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "num_train_epochs": 5,
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
    "eval_steps": 200,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "report_to": "wandb",
    "save_strategy": "steps",
    "save_steps": 200,
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "disable_tqdm": False
}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_dict = {}
    for metric in [ACCURACY, PRECISION, RECALL, F1]:
        metric_dict.update(metric.compute(predictions=predictions, references=labels))
    return metric_dict


def model_finetuning(model, tokenizer, fshandler, setup_function, use_artifacts, wandb_tags, 
                     wandb_group, log_model=False, params=None):
    
    if params is None:
        params = {
            "training": {}
        }

    os.environ["WANDB_LOG_MODEL"] = str(log_model).lower()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    model, support_set, test_set = setup_function(model, tokenizer, fshandler, params)
    
    run = wandb.init(project="aslan", tags=wandb_tags, job_type="training", group=wandb_group)

    num_labels = fshandler.intent_num
    wandb.config["support_size"] = len(support_set) / num_labels
    wandb.config["num_labels"] = num_labels

    for artifact in use_artifacts:
        run.use_artifact(artifact)
        run.use_artifact(artifact)

    settings = COMMON_ARGS.copy()
    settings.update(params["training"])

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

    if "separator" not in params:
        params["separator"] = "<sep>"
    if "test_size" not in params:
        params["test_size"] = None

    support_set = TokenizedDataset(support_ui, lambda x: x["text"] + params["separator"] + x["intent"], tokenizer)
    test_set = TokenizedDataset(test_ui, lambda x: x["text"] + params["separator"] + x["intent"],
                                tokenizer, sample_size=params["test_size"])

    return roberta, support_set, test_set


def setup_knn_roberta(roberta, tokenizer, fshandler, params):
    support_uu = UUDataset(fshandler.known, fshandler.known)
    test_uu = UUDataset(fshandler.unknown, fshandler.known)

    if "separator" not in params:
        params["separator"] = "<sep>"
    if "test_size" not in params:
        params["test_size"] = None

    support_set = TokenizedDataset(support_uu, lambda x: x["text_known"] + \
                                   params["separator"] + x["text_unknown"], tokenizer)
    test_set = TokenizedDataset(test_uu, lambda x: x["text_known"] + \
                                params["separator"] + x["text_unknown"], tokenizer, sample_size=params["test_size"])

    return roberta, support_set, test_set
