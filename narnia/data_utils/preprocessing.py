import wandb
import json
import pandas as pd
import numpy as np

from pathlib import Path
from transformers import BertTokenizerFast
from datasets import load_dataset, ClassLabel, load_metric

from .wandb_logging import log_clinc150


def refactor_clinc150():
    run = wandb.init(project="aslan", job_type="data-preprocessing",
                     notes="Refactor CLINC150 and create dataframes")
    run.use_artifact("CLINC150:v0")

    with open("data/oos-eval/data/data_full.json") as f:
        data_full = json.load(f)

    train = pd.DataFrame(data_full["train"]).rename(columns={0: "text", 1: "intent"})
    val = pd.DataFrame(data_full["val"]).rename(columns={0: "text", 1: "intent"})
    test = pd.DataFrame(data_full["test"]).rename(columns={0: "text", 1: "intent"})

    val = pd.concat([val, pd.DataFrame(data_full["oos_val"]).rename(columns={0: "text", 1: "intent"})])
    test = pd.concat([test, pd.DataFrame(data_full["oos_test"]).rename(columns={0: "text", 1: "intent"})])

    with open("data/oos-eval/data/domains.json") as f:
        domains = json.load(f)

    intent2domain = {"intent": ["oos"], "domain": ["oos"]}

    for domain, intents in domains.items():
        intent2domain["intent"] += intents
        intent2domain["domain"] += [domain] * len(intents)
        
    domains_df = pd.DataFrame(intent2domain).set_index("intent")

    train = train.join(domains_df, on="intent")
    val = val.join(domains_df, on="intent")
    test = test.join(domains_df, on="intent")

    train.to_csv("data/oos-eval/data/train.csv", index=False)
    val.to_csv("data/oos-eval/data/val.csv", index=False)
    test.to_csv("data/oos-eval/data/test.csv", index=False)
    domains_df.to_csv("data/oos-eval/data/domains.csv", index=False)

    log_clinc150(run)


def split_clinc150_domains():
    run = wandb.init(project="aslan", job_type="data-preprocessing",
                     notes="Split CLINC150 datasets into train, val and test domains.")
    run.use_artifact("CLINC150:v1")

    # Fix the error made in the last version of artifact
    with open("data/oos-eval/data/domains.json") as f:
        domains = json.load(f)
    intent2domain = {"intent": ["oos"], "domain": ["oos"]}
    for domain, intents in domains.items():
        intent2domain["intent"] += intents
        intent2domain["domain"] += [domain] * len(intents)
    domains_df = pd.DataFrame(intent2domain).set_index("intent")
    domains_df.to_csv("data/oos-eval/data/domains.csv")

    train = pd.read_csv("data/oos-eval/data/train.csv")
    val = pd.read_csv("data/oos-eval/data/val.csv")
    test = pd.read_csv("data/oos-eval/data/test.csv")

    train_domains = ["banking", "credit_cards", "kitchen_and_dining", "auto_and_commute", "utility", "meta"]
    val_domains = ["travel", "work"]
    test_domains = ["home", "small_talk"]

    dataset = {
        "train": {
            "train": train[train["domain"].isin(train_domains)],
            "val": val[val["domain"].isin(train_domains)],
            "test": test[test["domain"].isin(train_domains)]
        },
        "val": {
            "train": train[train["domain"].isin(val_domains)],
            "val": val[val["domain"].isin(val_domains)],
            "test": test[test["domain"].isin(val_domains)]
        },
        "test": {
            "train": train[train["domain"].isin(test_domains)],
            "val": val[val["domain"].isin(test_domains)],
            "test": test[test["domain"].isin(test_domains)]
        }
    }

    split_path = "data/oos-eval/data/split"
    Path(split_path).mkdir(parents=True, exist_ok=True)
    for dkey, dval in dataset.items():
        for skey, sval in dval.items():
            sval.to_csv(f"{split_path}/D{dkey}S{skey}.csv", index=False)

    log_clinc150(run)


def tokenize_clinc150():
    data_files = {}
    for d in ["train", "val", "test"]:
        for s in ["train", "val", "test"]:
            data_files[f"D{d}S{s}"] = f"artifacts/CLINC150:v3/split/D{d}S{s}.csv"

    raw = load_dataset("csv", data_files=data_files)

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    tokenized_dataset = raw.map(lambda examples: tokenizer(examples['text']), batched=True)

    cl = {}

    for dsplit in ["train", "val", "test"]:
        unique_intents = list(tokenized_dataset[f"D{dsplit}Strain"].unique("intent"))
        classlabel = ClassLabel(names=unique_intents)
        
        def encode_label(x):
            x["label"] = classlabel.str2int(x["intent"])
            return x

        for ssplit in ["train", "val", "test"]:
            print(dsplit, ssplit)
            tokenized_dataset[f"D{dsplit}S{ssplit}"] = tokenized_dataset[f"D{dsplit}S{ssplit}"].map(encode_label)

        cl[dsplit] = classlabel
        
    tokenized_dataset.save_to_disk("datasets/data")

    mapping = {}
    for dsplit in ["train", "val", "test"]:
        mapping[dsplit] = {
            "int2str": cl[dsplit]._int2str,
            "str2int": cl[dsplit]._str2int
        }
        
    with open("datasets/mapping.json", "w") as f:
        json.dump(mapping, f)

    run = wandb.init(project="aslan", job_type="data-preprocessing",
                    notes="Log tokenized and split CLINC150")

    my_data = wandb.Artifact("CLINC150-tokenized", type="dataset", description="Split and tokenized dataset for SLU from [here]"
                            "(https://github.com/clinc/oos-eval) in the form of huggingface datasets")
    run.use_artifact("CLINC150:latest")

    my_data.add_dir("datasets")
    run.log_artifact(my_data)

    wandb.finish()


def cut_indices(dset, support_size):
    np.random.seed(42)
    return list(map(int, list(pd.Series(dset["label"]).rename("label").reset_index(drop=False).groupby("label") \
                ["index"].apply(lambda x: np.random.choice(x, size=support_size)).explode().sample(frac=1))))


def add_label_idxs_to_clinc150():
    idxs = {}
    for k in range(1, 101):
        idxs[k] = {
            "DvalStrain": cut_indices(dataset["DvalStrain"], k),
            "DtestStrain": cut_indices(dataset["DtestStrain"], k),
        }

    with open("datasets/train_idxs.json", "w") as f:
        json.dump(idxs, f)

    run = wandb.init(project="aslan", job_type="data-preprocessing",
                    notes="Log tokenized and split CLINC150")

    my_data = wandb.Artifact("CLINC150-tokenized", type="dataset", description="Split and tokenized dataset for SLU from [here]"
                            "(https://github.com/clinc/oos-eval) in the form of huggingface datasets")
    run.use_artifact("CLINC150-tokenized:latest")

    my_data.add_dir("datasets")
    run.log_artifact(my_data)

    wandb.finish()

def split_clinc150_for_zero_shot():
    run = wandb.init(project="aslan", job_type="data-preprocessing",
                     notes="Split CLINC150 datasets into seen and unseen intents for zero shot")
    run.use_artifact("CLINC150:latest")

    train = pd.read_csv("data/oos-eval/data/train.csv")
    val = pd.read_csv("data/oos-eval/data/val.csv")
    test = pd.read_csv("data/oos-eval/data/test.csv")

    all_intents = train["intent"].unique().tolist()
    seen_intents = np.random.choice(all_intents, 112, replace=False).tolist()
    unseen_intents = list(set(all_intents) - set(seen_intents))

    metadata = {
        "seen_intents": seen_intents,
        "unseen_intents": unseen_intents
    }

    dataset = {
        "seen": {
            "train": train[train["intent"].isin(seen_intents)],
            "val": val[val["intent"].isin(seen_intents)],
            "test": test[test["intent"].isin(seen_intents)]
        },
        "unseen": {
            "train": train[train["intent"].isin(unseen_intents)],
            "val": val[val["intent"].isin(unseen_intents)],
            "test": test[test["intent"].isin(unseen_intents)]
        }
    }

    split_path = "data/oos-eval/data/zero_shot_split"
    Path(split_path).mkdir(parents=True, exist_ok=True)
    for dkey, dval in dataset.items():
        for skey, sval in dval.items():
            sval.to_csv(f"{split_path}/{dkey}_{skey}.csv", index=False)

    log_clinc150(run, metadata=metadata)
