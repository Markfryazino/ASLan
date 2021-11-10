import wandb
from typing import Dict, Callable, List, Tuple
import os
import logging

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from datasets import set_caching_enabled, ClassLabel

from environment import FewShotHandler, load_from_memory, set_generator, load_unseen, load_split_dataset
from few_shot_training import laboratory_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta, \
                              laboratory_pretraining, setup_pretraining_bert
from utils import set_random_seed, get_timestamp_str, append_prefix


set_caching_enabled(False)


def load_artifacts(artifacts: Dict[str, str], logger):
    logger("start loading artifacts")
    wandb.login()
    api = wandb.Api()

    for val in artifacts.values():
        api.artifact(f"broccoliman/aslan/{val}").download()

    logger("stop loading artifacts")


class FewShotLaboratory:
    def __init__(
        self,
        modules: List[Tuple[str, Callable[[FewShotHandler, Dict], Dict]]],
        pretraining_modules: List[Tuple[str, Callable[[Dict, Dict], Dict]]],
        artifacts: Dict[str, str],
        support_size: int = 10,
        logger: Callable[[str], None] = print,
        wandb_args: Dict = {},
        params: Dict = {},
        root_path: str = "artifacts/"
    ):
        self.config = {}
        self.modules = modules
        self.artifacts = artifacts
        self.support_size = support_size
        self.logger = logger
        self.wandb_args = wandb_args
        self.params = params
        self.root_path = root_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger(f"Initializing laboratory\nUsing device {self.device}")

        load_artifacts(artifacts, self.logger)
        self.seen, self.unseen, self.generator = None, None, None
        self.state = {"seen": self.seen}

    def init_data(self, dataset, split):
        self.logger(f"Setting dataset {dataset}, split {split}")
        self.seen, self.unseen = load_split_dataset(self.root_path, dataset, split)
        self.generator = set_generator(self.unseen, support_size=self.support_size)
        self.config.update({"dataset": dataset, "split": split})

    def run_series(self, dataset, random_states, use_negative_split=False):
        splits = list(range(5))
        if use_negative_split:
            splits.append(-1)

        logs = {}
        for split in splits:
            logs[split] = {}
            self.init_data(dataset, split)
            for state in random_states:
                self.logger(f"Starting run with split {split}, state {state}")
                metrics, handler = self.run(state)
                logs[split][state] = metrics

        self.logger("Finishing")
        wandb_run = wandb.init(**self.wandb_args, job_type="just-logging")

        for artifact in self.artifacts.values():
            wandb_run.use_artifact(f"broccoliman/aslan/{artifact}")

        wandb.log(logs)
        wandb.finish()

    def pretraining_run(self, random_state):
        wandb_run = wandb.init(**self.wandb_args, save_code=True)
        self.config["random_state"] = random_state

        for artifact in self.artifacts.values():
            wandb_run.use_artifact(f"broccoliman/aslan/{artifact}")

        self.logger(f"Starting pretraining run with random_state = {random_state}")  

        set_random_seed(random_state)     

        run_metrics = {}

        for block_name, block in self.pretraining_modules:
            self.logger(f"Running pretraining block {block_name}")

            if block_name not in self.params:
                self.params[block_name] = {}
            self.params[block_name]["block_name"] = block_name

            metrics = block(self.state, self.params[block_name])
            if len(metrics) > 0:
                final_metrics = {f"{block_name}/{key}": val for key, val in metrics.items()}
                wandb.log(final_metrics)

            run_metrics[block_name] = metrics
            self.logger(f"Pretraining block {block_name} finished. Metrics: {metrics}\n")

        self.logger(f"Pretraining run finished.")
        wandb.config.update(self.config)
        wandb.finish()

        return run_metrics, fshandler 

    def run(self, random_state):
        wandb_run = wandb.init(**self.wandb_args, save_code=True)
        self.config["random_state"] = random_state

        for artifact in self.artifacts.values():
            wandb_run.use_artifact(f"broccoliman/aslan/{artifact}")

        self.logger(f"Starting run with random_state = {random_state}")

        set_random_seed(random_state)
        known, unknown = next(self.generator)
        fshandler = FewShotHandler(unknown, known, device=self.device, logger=self.logger)

        run_metrics = {}

        for block_name, block in self.modules:
            self.logger(f"Running block {block_name}")

            if block_name not in self.params:
                self.params[block_name] = {}
            self.params[block_name]["block_name"] = block_name

            metrics = block(fshandler, self.params[block_name])
            if len(metrics) > 0:
                final_metrics = {f"{block_name}/{key}": val for key, val in metrics.items()}
                wandb.log(final_metrics)

            run_metrics[block_name] = metrics
            self.logger(f"Block {block_name} finished. Metrics: {metrics}\n")

        self.logger(f"Run finished.")
        wandb.config.update(self.config)
        wandb.finish()

        return run_metrics, fshandler


def load_knn_roberta(fshandler, params):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"})

    model = RobertaForSequenceClassification.from_pretrained(params["roberta_path"]).to(fshandler.device)
    model.resize_token_embeddings(len(tokenizer))
    fshandler.state["knn_roberta_model"] = model
    fshandler.state["knn_roberta_tokenizer"] = tokenizer
    return {}


def finetune_knn_roberta(fshandler, params):
    block_name = params["block_name"]
    del params["block_name"]
    model = fshandler.state["knn_roberta_model"]
    tokenizer = fshandler.state["knn_roberta_tokenizer"]

    model, metrics = laboratory_finetuning(model, tokenizer, fshandler, setup_knn_roberta, 
                                           prefix=block_name, params=params)
    fshandler.state["knn_roberta_model"] = model
    return metrics


def pretrain_naive_roberta(state, params):
    block_name = params["block_name"]
    del params["block_name"] 

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"}) 
    model = RobertaForSequenceClassification.from_pretrained('roberta-base').to(state["device"])
    model.resize_token_embeddings(len(tokenizer))

    model, metrics = laboratory_pretraining(model, tokenizer, state["seen_data"], setup_pretraining_bert, 
                                            prefix=block_name, params=params)
    state["naive_roberta_model"] = model
    state["naive_roberta_tokenizer"] = tokenizer

    if ("save_model" in params) and params["save_model"]:
        if "model_path" not in params:
            params["model_path"] = "./results/naive_roberta"

        model.save_pretrained(params["model_path"])
        tokenizer.save_pretrained(params["model_path"])

        my_data = wandb.Artifact(params["artifact_name"], type="model")
        my_data.add_dir(params["model_path"])
        run.log_artifact(my_data)

    return metrics


def encode_labels(state, params):
    unique_intents = list(state["seen_data"].unique("intent")["train"])
    classlabel = ClassLabel(names=unique_intents)
    state["seen_data"] = state["seen_data"].map(lambda x: {"label": classlabel.str2int(x["intent"]), **x})
    state["classlabel"] = classlabel
    return {}


def evaluate_knn_roberta(fshandler, params):
    settings = {key: val for key, val in params.items() if key in ["top_k", "batch_size"]}
    model = fshandler.state["knn_roberta_model"]
    tokenizer = fshandler.state["knn_roberta_tokenizer"]

    eval_result = fshandler.eval_stuu(model, tokenizer, **settings)
    if "verbose" not in params or not params["verbose"]:
        del eval_result["details"]
    return eval_result


def evaluate_pure_sbert(fshandler, params):
    sbert = None
    if "sbert" in params:
        sbert = params["sbert"]
    eval_result = fshandler.eval_pure_sbert(sbert=sbert)
    return eval_result
