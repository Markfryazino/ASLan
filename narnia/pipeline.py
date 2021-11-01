import wandb
from typing import Dict, Callable, List, Tuple
import os
import logging

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from environment import FewShotHandler, load_from_memory, set_generator
from few_shot_training import laboratory_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta
from utils import set_random_seed, get_timestamp_str, append_prefix


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
        artifacts: Dict[str, str],
        support_size: int = 10,
        logger: Callable[[str], None] = print,
        wandb_args: Dict = {},
        params: Dict = {}
    ):
        self.modules = modules
        self.artifacts = artifacts
        self.support_size = support_size
        self.logger = logger
        self.wandb_args = wandb_args
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger(f"Initializing laboratory\nUsing device {self.device}")

        load_artifacts(artifacts, self.logger)
        self.data, self.unseen = load_from_memory(f"artifacts/{artifacts['dataset']}")
        self.generator = set_generator(self.unseen, support_size=support_size)

    def run(self, random_state):
        wandb_run = wandb.init(**self.wandb_args, save_code=True)

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
