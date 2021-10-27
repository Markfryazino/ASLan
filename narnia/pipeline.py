import wandb
from typing import Dict
import os
import logging

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from environment import FewShotHandler, load_from_memory
from few_shot_training import model_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta
from utils import set_random_seed, append_prefix, get_timestamp_str


def load_artifacts(artifacts: Dict[str, str]):
    logging.debug("start loading artifacts")
    wandb.login()
    api = wandb.Api()

    for val in artifacts.values():
        api.artifact(f"broccoliman/aslan/{val}").download()

    logging.debug("stop loading artifacts")


def load_knn_roberta(roberta_path):
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"})

    model = RobertaForSequenceClassification.from_pretrained(roberta_path)
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer



def do_everything(
    artifacts: Dict[str, str],
    support_size: int = 10,
    random_state: int = 42,
    wandb_group: str = None
):
    if wandb_group is None:
        wandb_group = get_timestamp_str() + "Exp"
        logging.info(f"Wandb group is not set, setting it to {wandb_group}")

    load_artifacts(artifacts)
    data, unseen = load_from_memory(append_prefix(artifacts["dataset"]))

    set_random_seed(random_state)
    gen = set_generator(unseen, support_size=support_size)
    known, unknown = next(gen)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.debug(f"Using device {device}")
    fshandler = FewShotHandler(unknown, known, device=device)

    logging.debug("Loading model and tokenizer")
    model, tokenizer = load_knn_roberta(append_prefix(artifacts["knn_roberta"]))
    model.to(device)
    logging.debug("Model loaded")

    logging.debug("Starting finetuning")
    model = model_finetuning(model, tokenizer, fshandler, setup_knn_roberta, 
                             use_artifacts=artifacts.values(),
                             wandb_tags=["roberta", "clinc150", "knn-roberta"],
                             wandb_group=wandb_group,
                             params={"training": {"eval_steps": 500, "save_steps": 500}, "test_size": 0.2})
    logging.debug("Finetuning finished")

    logging.debug("Starting evaluation")
    eval_result = fshandler.eval_stuu(model, tokenizer, sbert, top_k=10, batch_size=128)
    logging.info(f"Evaluation finished. Accuracy: {eval_result['accuracy']}")

    logging.debug("Process finished.")
