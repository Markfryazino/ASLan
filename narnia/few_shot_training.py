import wandb
import numpy as np
import torch
import json
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, \
                         BertTokenizerFast, BertForSequenceClassification, \
                         RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorForLanguageModeling
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd

from environment import FewShotHandler, TokenizedDataset, UIDataset, UUDataset, STUUDataset, \
                        SBERTDataset, IEFormatDataset
from wandb_callback import WandbPrefixCallback, SBERTWandbCallback

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from generation import GENERATION_ARGS


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
    "evaluation_strategy": "epoch",
    "logging_dir": "./logs",
    "logging_steps": 10,
    "report_to": "wandb",
    "save_strategy": "epoch",
    "save_total_limit": 5,
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "disable_tqdm": False
}


def _compute_metrics(eval_pred, average="binary"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_dict = {}
    for metric in [ACCURACY, PRECISION, RECALL, F1]:
        try:
            metric_dict.update(metric.compute(predictions=predictions, references=labels, average=average))
        except:
            metric_dict.update(metric.compute(predictions=predictions, references=labels))
    return metric_dict


def compute_metrics(eval_pred):
    return _compute_metrics(eval_pred)


def compute_multiclass_metrics(eval_pred):
    return _compute_metrics(eval_pred, average="macro")


def model_finetuning(model, tokenizer, fshandler, setup_function, use_artifacts, wandb_tags, 
                     wandb_group, log_model=False, params=None):
    
    if params is None:
        params = {}

    if "training" not in params:
        params["training"] = {}

    os.environ["WANDB_LOG_MODEL"] = str(log_model).lower()

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    model, support_set, test_set = setup_function(model, tokenizer, fshandler, params)
    
    run = wandb.init(project="aslan", tags=wandb_tags, job_type="training", group=wandb_group)

    num_labels = fshandler.intent_num
    wandb.config["support_size"] = len(fshandler.known) / num_labels
    wandb.config["num_labels"] = num_labels

    for artifact in use_artifacts:
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


def laboratory_finetuning(model, tokenizer, fshandler, setup_function, prefix, log_model=False, params=None,
                          mode="binary"):
    if params is None:
        params = {}
    if "training" not in params:
        params["training"] = {}

    os.environ["WANDB_LOG_MODEL"] = str(log_model).lower()

    model, support_set, test_set = setup_function(model, tokenizer, fshandler, params)

    mode_args = {
        "multiclass": [COMMON_ARGS.copy(), compute_multiclass_metrics, 
                       DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")],
        "binary": [COMMON_ARGS.copy(), compute_metrics, 
                   DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")],
        "generation": [GENERATION_ARGS.copy(), None, DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)],
    }
    settings, computer, collator = mode_args[mode]
    settings.update(params["training"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings), 
        train_dataset=support_set,
        eval_dataset=test_set,
        data_collator=collator,
        compute_metrics=computer,
        callbacks=[WandbPrefixCallback(prefix)]
    )

    trainer.train()

    final_metrics = trainer.evaluate(support_set, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(test_set, metric_key_prefix="test"))

    return model, final_metrics


def laboratory_pretraining(model, tokenizer, seen_data, setup_function, prefix, log_model=False, params=None,
                           mode="multiclass"):
    if params is None:
        params = {}
    if "training" not in params:
        params["training"] = {}

    os.environ["WANDB_LOG_MODEL"] = str(log_model).lower()

    model, train_set, eval_set, test_set = setup_function(model, tokenizer, seen_data, params)

    mode_args = {
        "multiclass": [COMMON_ARGS.copy(), compute_multiclass_metrics, 
                       DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")],
        "binary": [COMMON_ARGS.copy(), compute_metrics, 
                   DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")],
        "generation": [GENERATION_ARGS.copy(), None, DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)],
    }
    settings, computer, collator = mode_args[mode]
    settings.update(params["training"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings), 
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collator,
        compute_metrics=computer,
        callbacks=[WandbPrefixCallback(prefix)]
    )

    trainer.train()

    final_metrics = trainer.evaluate(train_set, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(eval_set, metric_key_prefix="eval"))
    final_metrics.update(trainer.evaluate(test_set, metric_key_prefix="test"))

    return model, final_metrics


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


def setup_pretraining_bert(bert, tokenizer, seen_data, params=None):
    train_set = TokenizedDataset(seen_data["train"], lambda x: x["text"], tokenizer)
    eval_set = TokenizedDataset(seen_data["val"], lambda x: x["text"], tokenizer)
    test_set = TokenizedDataset(seen_data["test"], lambda x: x["text"], tokenizer)

    return bert, train_set, eval_set, test_set


def setup_pretraining_knn_roberta(roberta, tokenizer, seen_data, params=None):
    if params is None:
        params = {}
    if "dataset" not in params:
        params["dataset"] = {}

    train_raw = SBERTDataset(seen_data["train"], **params["dataset"])
    val_raw = SBERTDataset(seen_data["val"], **params["dataset"])
    test_raw = SBERTDataset(seen_data["test"], **params["dataset"])

    train = TokenizedDataset(train_raw, lambda x: x["source_text"] + "<sep>" + x["other_text"], tokenizer)
    val = TokenizedDataset(val_raw, lambda x: x["source_text"] + "<sep>" + x["other_text"], tokenizer)
    test = TokenizedDataset(test_raw, lambda x: x["source_text"] + "<sep>" + x["other_text"], tokenizer)
    return roberta, train, val, test


def setup_pretraining_naive_gpt2(gpt2, tokenizer, seen_data, params=None):
    train = TokenizedDataset(seen_data["train"], lambda x: "<start>" + x["intent"] + "<sep>" + x["text"] + "<end>",
                             tokenizer)
    val = TokenizedDataset(seen_data["val"], lambda x: "<start>" + x["intent"] + "<sep>" + x["text"] + "<end>",
                           tokenizer)
    test = TokenizedDataset(seen_data["test"], lambda x: "<start>" + x["intent"] + "<sep>" + x["text"] + "<end>",
                            tokenizer)
    return gpt2, train, val, test


def setup_pretraining_similarity_gpt2(gpt2, tokenizer, seen_data, params=None):
    def template(source, other, label):
        return f"<start>{source}<{label}>{other}<end>"

    if "dataset" not in params:
        params["dataset"] = {}

    raw_train = SBERTDataset(seen_data["train"], **params["dataset"])
    raw_val = SBERTDataset(seen_data["val"], **params["dataset"])
    raw_test = SBERTDataset(seen_data["test"], **params["dataset"])

    train = TokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"], x["label"], 
                             no_label=True), tokenizer)
    val = TokenizedDataset(raw_val, lambda x: template(x["source_text"], x["other_text"], x["label"], 
                           no_label=True), tokenizer)
    test = TokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"], x["label"], 
                            no_label=True), tokenizer)

    return gpt2, train, val, test


def setup_entailment_roberta(roberta, tokenizer, fshandler, params):
    support_ui = UIDataset(fshandler.known, fshandler.intents)
    test_ui = UIDataset(fshandler.unknown, fshandler.intents)

    if "separator" not in params:
        params["separator"] = "<sep>"
    if "test_size" not in params:
        params["test_size"] = None

    support_set = TokenizedDataset(support_ui, lambda x: x["text"] + params["separator"] + x["intent"], tokenizer)

    if "fake_data" in params:
        fake_data = params["fake_data"].map(lambda x: {"label": 0, "generated": x["generated"], \
                                                       "intent": x["intent"]}, load_from_cache_file=False)
        fake_data = TokenizedDataset(fake_data, lambda x: x["generated"] + params["separator"] + x["intent"],
                                     tokenizer)
        support_set = torch.utils.data.ConcatDataset([support_set, fake_data])

    test_set = TokenizedDataset(test_ui, lambda x: x["text"] + params["separator"] + x["intent"],
                                tokenizer, sample_size=params["test_size"])

    return roberta, support_set, test_set


def setup_knn_roberta(roberta, tokenizer, fshandler, params):
    support_uu, test_uu = None, None
    if "top_k" not in params:
        support_uu = UUDataset(fshandler.known, fshandler.known)
        test_uu = UUDataset(fshandler.known, fshandler.unknown)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support_uu = STUUDataset(fshandler.known, fshandler.known, top_k=params["top_k"], device=device)
        test_uu = STUUDataset(fshandler.known, fshandler.unknown, top_k=params["top_k"], device=device)

    if "separator" not in params:
        params["separator"] = "<sep>"
    if "test_size" not in params:
        params["test_size"] = None

    support_set = TokenizedDataset(support_uu, lambda x: x["text_known"] + \
                                   params["separator"] + x["text_unknown"], tokenizer)
    test_set = TokenizedDataset(test_uu, lambda x: x["text_known"] + \
                                params["separator"] + x["text_unknown"], tokenizer, sample_size=params["test_size"])

    return roberta, support_set, test_set


def sbert_training(model, train_data, prefix, eval_data=None, params=None):
    if params is None:
        params = {}
    if "training" not in params:
        params["training"] = {}
    if "dataset" not in params:
        params["dataset"] = {}
    if "batch_size" not in params:
        params["batch_size"] = 64

    train = IEFormatDataset(SBERTDataset(train_data, **params["dataset"]))
    evaluator = None

    if eval_data is not None:
        val = IEFormatDataset(SBERTDataset(eval_data, **params["dataset"]))
        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val, name='sbert-val')
    
    train_dataloader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=params["batch_size"])
    train_loss = losses.CosineSimilarityLoss(model=model)
    callback = SBERTWandbCallback(prefix)

    training_args = {
        "epochs": 5,
        "scheduler": "WarmupLinear",
        "output_path": "./results",
        "evaluation_steps": 100,
    }
    training_args.update(params["training"])

    model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          callback=callback.log,
          **training_args)

    return model, {"eval_score": evaluator(model)}
