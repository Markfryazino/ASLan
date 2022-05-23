import wandb
import os
import sys
import argparse


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

    return parser.parse_args()


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

    lab.init_data(f"SOAD:v2/CLINC150", -1)
    metrics, fshandler = lab.run(args.state)

    auger = Augmenter(fshandler.known, state=args.state, results_path=f"gen_results-size-{args.size}")

    SETTINGS = {
        "train_generator": {
            "t5_path": "../data/T0_3B",
            "num_train_epochs": 3,
            "prefix_length": 20,
            "sbert_path": "../data/all-mpnet-base-v2"
        },
        "generate": {
            "multiplier": 10,
        },
    }

    auger.train_generator(**SETTINGS["train_generator"])
    auger.generate(**SETTINGS["generate"])


if __name__ == "__main__":
    main()
