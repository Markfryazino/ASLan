import wandb
from typing import Dict, Callable, List, Tuple
from tqdm.auto import tqdm, trange
import os
import logging
from pathlib import Path

import torch
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification, \
                         GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling
from datasets import set_caching_enabled, ClassLabel, concatenate_datasets

from environment import FewShotHandler, load_from_memory, set_generator, load_unseen, load_split_dataset, \
                        STUUDataset, TokenizedDataset
from few_shot_training import laboratory_finetuning, setup_bert, setup_knn_roberta, setup_entailment_roberta, \
                              laboratory_pretraining, setup_pretraining_bert, sbert_training, \
                              setup_pretraining_knn_roberta, setup_pretraining_naive_gpt2, \
                              setup_pretraining_similarity_gpt2, setup_separate_gpt2
from utils import set_random_seed, get_timestamp_str, append_prefix, offline
from sentence_transformers import SentenceTransformer
from generation import gpt2_generate_fake_knowns, gpt2_generate_fake_similars

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


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
        extra_size: int = 0,
        val_size: int = 0,
        logger: Callable[[str], None] = print,
        wandb_args: Dict = {},
        params: Dict = {},
        root_path: str = "artifacts/"
    ):
        self.config = {
            "modules": [name for name, func in modules],
            "pretraining_modules": [name for name, func in pretraining_modules],
            "module_params": params,
            "support_size": support_size,
            "extra_size": extra_size,
            "val_size": val_size
        }
        self.modules = modules
        self.pretraining_modules = pretraining_modules
        self.artifacts = artifacts
        self.support_size = support_size
        self.extra_size = extra_size
        self.val_size = val_size
        self.logger = logger
        self.wandb_args = wandb_args
        self.params = params
        self.root_path = root_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        Path("./results").mkdir(parents=True, exist_ok=True)
        self.logger(f"Initializing laboratory\nUsing device {self.device}")

        if not offline():
            load_artifacts(artifacts, self.logger)
        else:
            os.environ["WANDB_MODE"] = "offline"

        self.seen, self.unseen, self.generator = None, None, None
        self.state = {
            "seen_data": self.seen,
            "device": self.device,
            "logger": self.logger
        }

    def init_data(self, dataset, split):
        self.logger(f"Setting dataset {dataset}, split {split}")
        self.seen, self.unseen = load_split_dataset(self.root_path, dataset, split)
        self.state["seen_data"] = self.seen
        self.generator = set_generator(self.unseen, support_size=self.support_size, extra_size=self.extra_size,
                                       val_size=self.val_size)
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

        return run_metrics

    def run(self, random_state):
        wandb_run = wandb.init(**self.wandb_args, save_code=True)
        self.config["random_state"] = random_state

        for artifact in self.artifacts.values():
            wandb_run.use_artifact(f"broccoliman/aslan/{artifact}")

        self.logger(f"Starting run with random_state = {random_state}")

        set_random_seed(random_state)

        set_data = next(self.generator)
        fshandler = FewShotHandler(self.support_size, set_data["test"], set_data["train"], device=self.device, 
                                   logger=self.logger, extra_known=set_data["extra"], val_known=set_data["val"])  
        fshandler.state.update(self.state)

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


def replace_knowns(fshandler, params):
    fshandler.replace_knowns()
    return {}


def load_knn_roberta(fshandler, params):
    if ("roberta_path" not in params) or (params["roberta_path"] is None):
        fshandler.log("Loading roberta from roberta-base checkpoint")
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"}) 
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(fshandler.device)
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(params["roberta_path"])
        model = RobertaForSequenceClassification.from_pretrained(params["roberta_path"]).to(fshandler.device)

    model.resize_token_embeddings(len(tokenizer))
    fshandler.state["knn_roberta_model"] = model
    fshandler.state["knn_roberta_tokenizer"] = tokenizer
    return {}


def load_gpt2(fshandler, params):
    if ("gpt2_path" not in params) or (params["gpt2_path"] is None):
        fshandler.log("Loading gpt-2 from gpt2 base checkpoint")
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True, padding=True)
        tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>",
                                    "eos_token": "<end>", "unk_token": "<unk>"})
        model = GPT2LMHeadModel.from_pretrained("gpt2")
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained(params["gpt2_path"])
        model = GPT2LMHeadModel.from_pretrained(params["gpt2_path"])
    model.resize_token_embeddings(len(tokenizer))
    fshandler.state["gpt2_model"] = model
    fshandler.state["gpt2_tokenizer"] = tokenizer
    return {}


def load_sbert(fshandler, params):
    model = SentenceTransformer(params["sbert_path"]).to(fshandler.device)
    fshandler.state["sbert"] = model
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


def save_model(state, params):
    model = state[params["model"]]
    tokenizer = state[params["tokenizer"]]

    model.save_pretrained(params["model_path"])
    tokenizer.save_pretrained(params["model_path"])

    my_data = wandb.Artifact(params["artifact_name"], type="model")
    my_data.add_dir(params["model_path"])
    wandb.log_artifact(my_data)

    return {}

def save_sbert(state, params):
    my_data = wandb.Artifact(params["artifact_name"], type="model")
    my_data.add_dir(params["sbert_path"])
    wandb.log_artifact(my_data)
    return {}


def pretrain_naive_roberta(state, params):
    block_name = params["block_name"]
    del params["block_name"] 

    num_labels = len(state["seen_data"]["train"].unique("label"))

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"}) 
    model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                             num_labels=num_labels).to(state["device"])
    model.resize_token_embeddings(len(tokenizer))

    model, metrics = laboratory_pretraining(model, tokenizer, state["seen_data"], setup_pretraining_bert, 
                                            prefix=block_name, params=params)
    state["naive_roberta_model"] = model
    state["naive_roberta_tokenizer"] = tokenizer

    return metrics


def pretrain_sbert(state, params):
    block_name = params["block_name"]
    del params["block_name"]

    sbert = SentenceTransformer("all-mpnet-base-v2").to(state["device"])
    sbert, metrics = sbert_training(sbert, state["seen_data"]["train"], prefix=block_name, 
                                    eval_data=state["seen_data"]["val"], params=params)
    state["sbert"] = sbert
    return metrics


def pretrain_naive_gpt2(state, params):
    block_name = params["block_name"]
    del params["block_name"]

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True, padding=True)
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>",
                                  "eos_token": "<end>", "unk_token": "<unk>"})
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    model, metrics = laboratory_pretraining(model, tokenizer, state["seen_data"], setup_pretraining_naive_gpt2, 
                                            prefix=block_name, params=params, mode="generation")

    state["naive_gpt2_model"] = model
    state["naive_gpt2_tokenizer"] = tokenizer

    return metrics


def pretrain_similarity_gpt2(state, params):
    block_name = params["block_name"]
    del params["block_name"]

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', truncation=True, padding=True)
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>", "bos_token": "<start>",
                                  "eos_token": "<end>", "unk_token": "<unk>"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["<0>", "<1>"]})
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))

    model, metrics = laboratory_pretraining(model, tokenizer, state["seen_data"], setup_pretraining_similarity_gpt2, 
                                            prefix=block_name, params=params, mode="generation")

    state["similarity_gpt2_model"] = model
    state["similarity_gpt2_tokenizer"] = tokenizer

    return metrics


def finetune_separate_gpt2(fshandler, params):
    block_name = params["block_name"]
    del params["block_name"]

    model = fshandler.state["gpt2_model"]
    tokenizer = fshandler.state["gpt2_tokenizer"]

    model, metrics = laboratory_finetuning(model, tokenizer, fshandler, setup_separate_gpt2, 
                                           prefix=block_name, params=params, mode="generation")

    fshandler.state[params["prefix"] + "_gpt2_model"] = model
    fshandler.state[params["prefix"] + "_gpt2_tokenizer"] = tokenizer

    return metrics


def finetune_sbert(fshandler, params):
    block_name = params["block_name"]
    del params["block_name"]

    sbert = fshandler.state["sbert"].to(fshandler.state["device"])
    sbert, metrics = sbert_training(sbert, fshandler.known, prefix=block_name, 
                                    eval_data=fshandler.val_known, params=params)
    fshandler.state["sbert"] = sbert
    return metrics


def pretrain_knn_roberta(state, params):
    block_name = params["block_name"]
    del params["block_name"]

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    tokenizer.add_special_tokens({"sep_token": "<sep>", "pad_token": "<pad>"}) 
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2).to(state["device"])
    model.resize_token_embeddings(len(tokenizer))

    model, metrics = laboratory_pretraining(model, tokenizer, state["seen_data"], setup_pretraining_knn_roberta, 
                                            prefix=block_name, params=params)
    state["knn_roberta_model"] = model
    state["knn_roberta_tokenizer"] = tokenizer

    return metrics


def encode_labels(state, params):
    unique_intents = list(state["seen_data"].unique("intent")["train"])
    classlabel = ClassLabel(names=unique_intents)
    state["seen_data"] = state["seen_data"].map(lambda x: {"label": classlabel.str2int(x["intent"]), **x},
                                                load_from_cache_file=False)
    state["classlabel"] = classlabel
    return {}


def encode_fshandler_labels(fshandler, params):
    unique_intents = list(fshandler.known.unique("intent"))
    classlabel = ClassLabel(names=unique_intents)
    fshandler.known = fshandler.known.map(lambda x: {"label": classlabel.str2int(x["intent"]), **x},
                                          load_from_cache_file=False)
    fshandler.unknown = fshandler.unknown.map(lambda x: {"label": classlabel.str2int(x["intent"]), **x},
                                              load_from_cache_file=False)

    if fshandler.extra_known is not None:
        fshandler.extra_known = fshandler.extra_known.map(lambda x: {"label": classlabel.str2int(x["intent"]), **x},
                                                        load_from_cache_file=False)

    if fshandler.val_known is not None:
        fshandler.val_known = fshandler.val_known.map(lambda x: {"label": classlabel.str2int(x["intent"]), **x},
                                                      load_from_cache_file=False)

    fshandler.state["classlabel"] = classlabel
    return {}


def evaluate_knn_roberta(fshandler, params):
    settings = {key: val for key, val in params.items() if key in ["top_k", "batch_size"]}
    model = fshandler.state["knn_roberta_model"]
    tokenizer = fshandler.state["knn_roberta_tokenizer"]

    fakes = None
    if ("use_fakes" in params) and params["use_fakes"]:
        fakes = fshandler.state["fake_known"]

    eval_result = fshandler.eval_stuu(model, tokenizer, fakes, **settings)
    if "verbose" not in params or not params["verbose"]:
        del eval_result["details"]

    return eval_result


def evaluate_pure_sbert(fshandler, params):
    sbert = None
    if "sbert" in params:
        sbert = params["sbert"]
    eval_result = fshandler.eval_pure_sbert(sbert=sbert)
    return eval_result


def synthesize_fake_knowns(fshandler, params):
    intent_size = 10
    if "intent_size" in params:
        intent_size = params["intent_size"]
    fake_dataset = gpt2_generate_fake_knowns(fshandler.state["gpt2_model"], fshandler.state["gpt2_tokenizer"],
                                             fshandler.unknown.unique("intent"), intent_size)
    fshandler.state["fake_known"] = fake_dataset
    return {}


def synthesize_fake_similar(fshandler, params):
    example_size = 1
    if "example_size" in params:
        example_size = params["example_size"]

    settings = None
    if "generation" in params:
        settings = params["generation"]

    fake_positives = gpt2_generate_fake_similars(fshandler.state["positive_gpt2_model"], 
                                                 fshandler.state["positive_gpt2_tokenizer"],
                                                 fshandler.known["text"], fshandler.known["intent"],
                                                 example_size, 1, settings)
    fake_negatives = gpt2_generate_fake_similars(fshandler.state["negative_gpt2_model"], 
                                                 fshandler.state["negative_gpt2_tokenizer"],
                                                 fshandler.known["text"], fshandler.known["intent"],
                                                 example_size, 0, settings)
    fshandler.state["fake_similar"] = concatenate_datasets([fake_positives, fake_negatives])
    return {}


def eval_gpt2(fshandler, params):
    def template(source, other):
        return f"<start>{source}<sep>{other}<end>"

    prefix = params["prefix"]
    model = fshandler.state[prefix + "_gpt2_model"]
    tokenizer = fshandler.state[prefix + "_gpt2_tokenizer"]

    fshandler.log("Building STUU dataset for evaluating GPT-2")

    stuu = STUUDataset(fshandler.known, fshandler.val_known, top_k=params["top_k"])
    positive = [el for el in stuu if el["label"] == 1]
    negative = [el for el in stuu if el["label"] == 0]

    tok_positive = TokenizedDataset(positive, lambda x: template(x["text_known"], x["text_unknown"]), tokenizer, no_label=True)
    tok_negative = TokenizedDataset(negative, lambda x: template(x["text_known"], x["text_unknown"]), tokenizer, no_label=True)

    collator = DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
    positive_loader = torch.utils.data.DataLoader(tok_positive, batch_size=1, collate_fn=collator)
    negative_loader = torch.utils.data.DataLoader(tok_negative, batch_size=1, collate_fn=collator)

    positive_losses = []
    negative_losses = []

    fshandler.log("Predicting perplexities")

    with torch.inference_mode():
        for batch in tqdm(positive_loader):
            positive_losses.append(model(**batch.to("cuda")).loss.item())

        for batch in tqdm(negative_loader):
            negative_losses.append(model(**batch.to("cuda")).loss.item())

    positive_losses = np.array(positive_losses)
    negative_losses = np.array(negative_losses)

    ax = sns.violinplot(data=[positive_losses, negative_losses])
    ax.set_ylim(0, 7)
    ax.set_xticklabels(["positive", "negative"])
    ax.set_ylabel("loss")

    metrics = {
        "mean_positive_loss": positive_losses.mean(),
        "mean_negative_loss": negative_losses.mean(),
        "mean_losses_difference": positive_losses.mean() - negative_losses.mean(),
        "median_positive_loss": np.median(positive_losses),
        "median_negative_loss": np.median(negative_losses),
        "median_losses_difference": np.median(positive_losses) - np.median(negative_losses),
        "plot": ax      
    }

    return metrics