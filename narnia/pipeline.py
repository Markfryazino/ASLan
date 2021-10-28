import wandb
from typing import Dict, Callable, List, Tuple
import os
import logging

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from environment import FewShotHandler, load_from_memory, set_generator
from few_shot_training import laboratory_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta
from utils import set_random_seed, append_prefix, get_timestamp_str


def load_artifacts(artifacts: Dict[str, str]):
    logging.log(LOGGING_LEVEL, "start loading artifacts")
    wandb.login()
    api = wandb.Api()

    for val in artifacts.values():
        api.artifact(f"broccoliman/aslan/{val}").download()

    logging.log(LOGGING_LEVEL, "stop loading artifacts")


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
        self.artifact = artifacts
        self.modules = modules
        self.support_size = support_size
        self.logger = logger
        self.wandb_args = wandb_args
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger(f"Initializing laboratory\nUsing device {self.device}")

        load_artifacts(artifacts)
        self.data, self.unseen = load_from_memory(append_prefix(artifacts["dataset"]))
        self.generator = set_generator(self.unseen, support_size=support_size)

    def run(self, random_state):
        wandb_run = wandb.init(**self.wandb_args, save_code=True)

        for artifact in self.artifacts.values():
            wandb_run.use_artifact(artifact)

        self.logger(f"Starting run with random_state = {random_state}")

        set_random_seed(random_state)
        known, unknown = next(self.generator)
        fshandler = FewShotHandler(unknown, known, device=self.device, logger=self.logger)

        for block_name, block in self.modules:
            self.logger(f"Running block {block_name}")

            if block_name not in self.params:
                self.params[block_name] = {}
            self.params[block_name]["block_name"] = block_name

            metrics = block(fshandler, self.params[block_name])
            if len(metrics) > 0:
                final_metrics = {f"{block_name}/{key}": val for key, val in metrics.items()}
                wandb.log(final_metrics)
            self.logger(f"Block {block_name} finished. Metrics: {metrics}")

        self.logger(f"Run finished.")
        wandb.finish()


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
    settings = params.copy()
    del settings["block_name"]

    model, metrics = laboratory_finetuning(model, tokenizer, fshandler, setup_knn_roberta, 
                                           prefix=params["block_name"], params=settings)
    fshandler.state["knn_roberta_model"] = model
    return metrics


def evaluate_knn_roberta(fshandler, params):
    eval_result = fshandler.eval_stuu(model, tokenizer, top_k=10, batch_size=128)
    if "verbose" not in params or not params["verbose"]:
        del eval_result["details"]
    return eval_result


def do_everything(
    artifacts: Dict[str, str],
    support_size: int = 10,
    random_state: int = 42,
    wandb_group: str = None,
    logging_level: int = logging.INFO
):
    LOGGING_LEVEL = logging_level

    if wandb_group is None:
        wandb_group = get_timestamp_str() + "Exp"
        logging.log(LOGGING_LEVEL, f"Wandb group is not set, setting it to {wandb_group}")

    load_artifacts(artifacts)
    data, unseen = load_from_memory(append_prefix(artifacts["dataset"]))

    set_random_seed(random_state)
    gen = set_generator(unseen, support_size=support_size)
    known, unknown = next(gen)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.log(LOGGING_LEVEL, f"Using device {device}")
    fshandler = FewShotHandler(unknown, known, device=device)

    logging.log(LOGGING_LEVEL, "Loading model and tokenizer")
    model, tokenizer = load_knn_roberta(append_prefix(artifacts["knn_roberta"]))
    model.to(device)
    logging.log(LOGGING_LEVEL, "Model loaded")

    logging.log(LOGGING_LEVEL, "Starting finetuning")
    model = model_finetuning(model, tokenizer, fshandler, setup_knn_roberta, 
                             use_artifacts=artifacts.values(),
                             wandb_tags=["roberta", "clinc150", "knn-roberta"],
                             wandb_group=wandb_group,
                             params={"test_size": 0.3, "top_k": 10})
    logging.log(LOGGING_LEVEL, "Finetuning finished")

    logging.log(LOGGING_LEVEL, "Starting evaluation")
    eval_result = fshandler.eval_stuu(model, tokenizer, top_k=10, batch_size=128)
    logging.log(LOGGING_LEVEL, f"Evaluation finished. Accuracy: {eval_result['accuracy']}")

    logging.log(LOGGING_LEVEL, "Process finished.")
