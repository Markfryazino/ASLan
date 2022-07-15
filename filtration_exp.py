import wandb
import os
import sys
import argparse
import json

from datasets import Dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--state",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--out_size",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--fake_path",
        type=str,
        required=True,
    )

    return parser.parse_args()


def get_fakes(path, fshandler, state):
    with open(path + f"/{state}-fakes.json") as f:
        fakes = json.load(f)

    texts = fshandler.known["text"]

    good_intents = []
    good_texts = []

    for fake in fakes:
        if fake["fake_text"] in texts:
            continue
        texts.append(fake["fake_text"])
        good_intents.append(fake["intent"])
        good_texts.append(fake["fake_text"])

    fake_dataset = Dataset.from_dict({"intent": good_intents, "text": good_texts})
    return fake_dataset


def main():
    args = parse_args()

    os.environ["OFFLINE"] = "True"
    os.environ["WANDB_MODE"] = "offline"
    sys.path.append("narnia")

    from pipeline import FewShotLaboratory
    from augmenter import Augmenter

    wandb.login()

    lab = FewShotLaboratory(
        modules=[],
        pretraining_modules=[],
        artifacts={"dataset": "SOAD:v2"},
        support_size=args.size,
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

    lab.init_data(f"SOAD:v2/HWU64", -1)
    metrics, fshandler = lab.run(args.state)

    auger = Augmenter(fshandler.known, state=args.state,
                      results_path=f"filtered-size-{args.size}to{args.out_size}-hwu64")
    auger.fake_dataset = get_fakes(args.fake_path, fshandler, args.state)

    SETTINGS = {
        "train_corrector": {
            "model": "../data/roberta-base",
            "training_args": {
                "num_train_epochs": 25,
            }
        },
        "correct": {
            "threshold": 0.3
        },
        "diversify": {
            "model_type": "../data/roberta-base",
            "intent_size": args.out_size,
            "num_layers": 10
        }
    }

    print("start training corrector")
    auger.train_corrector(**SETTINGS["train_corrector"])
    print("start correcting")
    auger.correct(**SETTINGS["correct"])
    print("start diversifying")
    auger.diversify(**SETTINGS["diversify"])
    print("ok")


if __name__ == "__main__":
    main()
