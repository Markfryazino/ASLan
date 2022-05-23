import torch
import wandb
import os
import sys

os.environ["OFFLINE"] = "True"
os.environ["WANDB_MODE"] = "offline"
sys.path.append("narnia")

from pipeline import FewShotLaboratory
from augmenter import Augmenter

wandb.login()

SETTINGS = {
    "dataset": "CLINC150",
    "random_state": 13,
    "support_size": 10,
    "train_generator": {
        "t5_path": "../data/T0_3B",
        "num_train_epochs": 5,
        "prefix_length": 30,
        "batch_size": 4,
        "sbert_path": "../data/all-mpnet-base-v2"
    },
    "train_corrector": {
        "model": "../data/roberta-base",
        "training_args": {
            "num_train_epochs": 25,
        }
    },
    "generate": {
        "multiplier": 25,
        "generation_args": {
            "top_k": None,
            "temperature": 0.8,
            "top_p": 0.9
        }
    },
    "correct": {
        "threshold": 0.3
    },
    "diversify": {
        "model_type": "../data/roberta-base",
        "intent_size": 10,
        "num_layers": 10
    }
}

PARAMETER_RANGE = {
    "dataset": ["CLINC150", "BANKING77", "HWU64", "SNIPS", "ATIS"],
    "random_state": [0, 1, 2, 3, 4],
    "generate": {
        "generation_args": {
            "top_k": None,
            #"temperature": [0.5, 0.8, 1.0],
            #"top_p": [0.8, 0.9, 0.95]
            "temperature": [0.8],
            "top_p": [0.9]
        }
    },
    "correct": {
        "threshold": [0.2, 0.3, 0.4]
    },
    "diversify": {
        "intent_size": [5, 10, 20, 50]
    }
}

for random_state in PARAMETER_RANGE["random_state"]:
    for dataset in PARAMETER_RANGE["dataset"]: 
        SETTINGS["random_state"] = random_state
        SETTINGS["dataset"] = dataset

        lab = FewShotLaboratory(
            modules=[],
            pretraining_modules=[],
            artifacts={"dataset": "SOAD:v2"},
            support_size=SETTINGS["support_size"],
            extra_size=0,
            val_size=0,
            logger=print,
            wandb_args={
                "project": "aslan",
                "entity": "broccoliman",
                "job_type": "loading",
                "tags": ["just-load"]
            },
            params={},
            root_path="../data")

        lab.init_data(f"SOAD:v2/{SETTINGS['dataset']}", -1)
        metrics, fshandler = lab.run(SETTINGS["random_state"])

        auger = Augmenter(fshandler.known)

        auger.train_generator(**SETTINGS["train_generator"])
        auger.train_corrector(**SETTINGS["train_corrector"])

        for temperature in PARAMETER_RANGE["generate"]["generation_args"]["temperature"]:
            for top_p in PARAMETER_RANGE["generate"]["generation_args"]["top_p"]:
                SETTINGS["generate"]["generation_args"]["temperature"] = temperature
                SETTINGS["generate"]["generation_args"]["top_p"] = top_p

                auger.generate(**SETTINGS["generate"])

                for threshold in PARAMETER_RANGE["correct"]["threshold"]:
                    SETTINGS["correct"]["threshold"] = threshold
                    auger.correct(**SETTINGS["correct"])

                for intent_size in PARAMETER_RANGE["diversify"]["intent_size"]:
                    auger.diversify(**SETTINGS["diversify"])
