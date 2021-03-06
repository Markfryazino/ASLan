import wandb
import numpy as np
import torch
import json
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding, \
                         BertTokenizerFast, BertForSequenceClassification, \
                         RobertaTokenizerFast, RobertaForSequenceClassification, DataCollatorForLanguageModeling, \
                         DataCollatorForSeq2Seq
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd

from environment import FewShotHandler, TokenizedDataset, UIDataset, UUDataset, STUUDataset, \
                        SBERTDataset, IEFormatDataset, SortingDataset, CurriculumIterableDataset, \
                        IterableTokenizedDataset, T5TokenizedDataset, IterableT5TokenizedDataset
from wandb_callback import WandbPrefixCallback, SBERTWandbCallback

from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from generation import GENERATION_ARGS
from utils import offline

if not offline():
    ACCURACY = load_metric("accuracy")
    PRECISION = load_metric("precision")
    RECALL = load_metric("recall")
    F1 = load_metric("f1")
else:
    ACCURACY = load_metric("../portal/metrics/accuracy.py")
    PRECISION = load_metric("../portal/metrics/precision.py")
    RECALL = load_metric("../portal/metrics/recall.py")
    F1 = load_metric("../portal/metrics/f1.py") 

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
        "seq2seq": [GENERATION_ARGS.copy(), None, DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)],
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

    final_metrics = {}
    if "evaluate_on_train" in params and not params["evaluate_on_train"]:
        final_metrics.update(trainer.evaluate(support_set, metric_key_prefix="train"))
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

    train = TokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"], x["label"]), tokenizer,
                             no_label=True)
    val = TokenizedDataset(raw_val, lambda x: template(x["source_text"], x["other_text"], x["label"]), tokenizer,
                           no_label=True)
    test = TokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"], x["label"]), tokenizer,
                            no_label=True)

    return gpt2, train, val, test


def setup_separate_gpt2(gpt2, tokenizer, fshandler, params):
    def template(source, other, intent=None):
        if intent is not None:
            return f"<start>{intent}<sep>{source}<sep>{other}<end>"
        else:
            return f"<start>{source}<sep>{other}<end>"

    if "curriculum" not in params or not params["curriculum"]:
        if "test_size" not in params:
            params["test_size"] = None

        raw_train = SBERTDataset(fshandler.known, **params["dataset"])
        raw_test = SBERTDataset(fshandler.val_known, **params["dataset"])

        if "add_intent" in params and params["add_intent"]:
            train = TokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"], 
                                     x["source_intent"]), tokenizer, no_label=True)
            test = TokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"],
                                    x["source_intent"]), tokenizer, no_label=True, sample_size=params["test_size"])
        else:
            train = TokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"]), tokenizer, 
                                    no_label=True)
            test = TokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"]), tokenizer, 
                                    no_label=True, sample_size=params["test_size"])          

    else:
        if "curriculum_dataset" not in params:
            params["curriculum_dataset"] = {}

        raw_train = CurriculumIterableDataset(
            SortingDataset(fshandler.known, **params["train_dataset"]),
            **params["curriculum_dataset"]
        )
        raw_test = SBERTDataset(fshandler.val_known, **params["val_dataset"])

        if "add_intent" in params and params["add_intent"]:
            train = IterableTokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"],
                                            x["source_intent"]), tokenizer, no_label=True)
            test = IterableTokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"],
                                            x["source_intent"]), tokenizer, no_label=True)
        else:
            train = IterableTokenizedDataset(raw_train, lambda x: template(x["source_text"], x["other_text"]), 
                                            tokenizer, no_label=True)
            test = IterableTokenizedDataset(raw_test, lambda x: template(x["source_text"], x["other_text"]), 
                                            tokenizer, no_label=True)
        
    return gpt2, train, test


def setup_separate_t5(t5, tokenizer, fshandler, params):
    def template_encoder(source, intent=None):
        if intent is not None:
            return f"{intent}<sep>{source}"
        else:
            return source
    def template_decoder(other):
        return f"<start>{other}<end>"

    if "curriculum" not in params or not params["curriculum"]:
        if "test_size" not in params:
            params["test_size"] = None

        raw_train = SBERTDataset(fshandler.known, **params["dataset"])
        raw_test = SBERTDataset(fshandler.val_known, **params["dataset"])

        if "add_intent" in params and params["add_intent"]:
            train = T5TokenizedDataset(raw_train, lambda x: template_encoder(x["source_text"], x["source_intent"]), 
                                       lambda x: template_decoder(x["other_text"]), tokenizer)
            test = T5TokenizedDataset(raw_test, lambda x: template_encoder(x["source_text"], x["source_intent"]),
                                      lambda x: template_decoder(x["other_text"]), tokenizer,
                                      sample_size=params["test_size"])
        else:
            train = T5TokenizedDataset(raw_train, lambda x: template_encoder(x["source_text"]), 
                                       lambda x: template_decoder(x["other_text"]), tokenizer)
            test = T5TokenizedDataset(raw_test, lambda x: template_encoder(x["source_text"]), 
                                      lambda x: template_decoder(x["other_text"]), tokenizer, 
                                      sample_size=params["test_size"])          

    else:
        if "curriculum_dataset" not in params:
            params["curriculum_dataset"] = {}

        raw_train = CurriculumIterableDataset(
            SortingDataset(fshandler.known, **params["train_dataset"]),
            **params["curriculum_dataset"]
        )
        raw_test = SBERTDataset(fshandler.val_known, **params["val_dataset"])

        if "add_intent" in params and params["add_intent"]:
            train = IterableT5TokenizedDataset(raw_train, lambda x: template_encoder(x["source_text"], 
                                               x["source_intent"]), lambda x: template_decoder(x["other_text"]), 
                                               tokenizer)
            test = IterableT5TokenizedDataset(raw_test, lambda x: template_encoder(x["source_text"], x["source_intent"]),
                                              lambda x: template_decoder(x["other_text"]), tokenizer)
        else:
            train = IterableT5TokenizedDataset(raw_train, lambda x: template_encoder(x["source_text"]), 
                                               lambda x: template_decoder(x["other_text"]), tokenizer)
            test = IterableT5TokenizedDataset(raw_test, lambda x: template_encoder(x["source_text"]), 
                                              lambda x: template_decoder(x["other_text"]), tokenizer)  
        
    return t5, train, test


def setup_entailment_roberta(roberta, tokenizer, fshandler, params):
    support_ui = UIDataset(fshandler.known, fshandler.intents)
    test_ui = UIDataset(fshandler.val_known, fshandler.intents)

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

    sbert = None
    if "sbert" in fshandler.state:
        fshandler.log("found sbert in state, using it")
        sbert = fshandler.state["sbert"]

    if "top_k" not in params:
        support_uu = UUDataset(fshandler.known, fshandler.known)
        test_uu = UUDataset(fshandler.known, fshandler.val_known)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if "pair_numbers" not in params:
            pair_numbers = {
                "hard_positive": fshandler.support_size,
                "hard_negative": fshandler.support_size,
                "easy_positive": 0,
                "easy_negative": 0
            }
        else:
            pair_numbers = params["pair_numbers"]

        support_uu = SBERTDataset(fshandler.known, sbert=sbert, logger=fshandler.log, pair_numbers=pair_numbers,
                                  device=device)
        test_uu = STUUDataset(fshandler.known, fshandler.val_known, sbert=sbert, 
                              top_k=params["top_k"], device=device)

    if ("use_fakes" in params) and params["use_fakes"] and ("fake_similar" in fshandler.state):
        fshandler.log(f"Using fake data for finetuning, concatenating {len(support_uu)}" + \
                      f" + {len(fshandler.state['fake_similar'])} examples")
        support_uu = torch.utils.data.ConcatDataset([support_uu, fshandler.state["fake_similar"]])

    if "separator" not in params:
        params["separator"] = "<sep>"
    if "test_size" not in params:
        params["test_size"] = None

    if "add_intent" in params and params["add_intent"]:
        support_set = TokenizedDataset(support_uu, lambda x: x["source_intent"] + params["separator"] + \
                                       x["source_text"] + params["separator"] + x["other_text"], tokenizer)
        test_set = TokenizedDataset(test_uu, lambda x: x["intent_known"] + params["separator"] + x["text_known"] + \
                                    params["separator"] + x["text_unknown"], tokenizer,
                                    sample_size=params["test_size"])
    else:
        support_set = TokenizedDataset(support_uu, lambda x: x["source_text"] + \
                                    params["separator"] + x["other_text"], tokenizer)
        test_set = TokenizedDataset(test_uu, lambda x: x["text_known"] + \
                                    params["separator"] + x["text_unknown"], tokenizer,
                                    sample_size=params["test_size"])

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


def naive_finetuning(model, tokenizer, known, val_known=None, unknown=None, wandb_args=None, params=None):
    if params is None:
        params = {}

    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    run = wandb.init(**wandb_args)

    settings = COMMON_ARGS.copy()
    settings.update(params)

    train = TokenizedDataset(known, lambda x: x["text"], tokenizer)
    val = None if val_known is None else TokenizedDataset(val_known, lambda x: x["text"], tokenizer) 
    test = None if unknown is None else TokenizedDataset(unknown, lambda x: x["text"], tokenizer)

    if val is None:
        settings["evaluation_strategy"] = "no"
        settings["save_strategy"] = "no"

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**settings), 
        train_dataset=train,
        eval_dataset=val,
        data_collator=collator,
        compute_metrics=compute_multiclass_metrics
    )

    trainer.train()

    final_metrics = trainer.evaluate(train, metric_key_prefix="train")

    if val is not None:
        final_metrics.update(trainer.evaluate(val, metric_key_prefix="eval"))

    if test is not None:
        final_metrics.update(trainer.evaluate(test, metric_key_prefix="test"))

    wandb.log(final_metrics)

    run.finish()
    return model, trainer
