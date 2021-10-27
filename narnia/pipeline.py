import wandb
from typing import Dict
import os
import logging

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

from environment import FewShotHandler, load_from_memory, set_generator
from few_shot_training import model_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta
from utils import set_random_seed, append_prefix, get_timestamp_str, LOGGING_LEVEL


def load_artifacts(artifacts: Dict[str, str]):
    logging.log(LOGGING_LEVEL, "start loading artifacts")
    wandb.login()
    api = wandb.Api()

    for val in artifacts.values():
        api.artifact(f"broccoliman/aslan/{val}").download()

    logging.log(LOGGING_LEVEL, "stop loading artifacts")


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
