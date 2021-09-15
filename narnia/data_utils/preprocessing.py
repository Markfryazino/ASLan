import wandb
import json
import pandas as pd

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
