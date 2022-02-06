from datasets import set_caching_enabled
set_caching_enabled(False)

import sys
sys.path.append("../narnia")

from pipeline import encode_fshandler_labels, load_knn_roberta, finetune_knn_roberta, evaluate_knn_roberta, \
    FewShotLaboratory

import os
os.environ["OFFLINE"] = "True"
os.environ["WANDB_MODE"] = "offline"

BATCH_SIZE = 128
POSSIBLE_SIZES = [5, 10, 20, 30]
RANDOM_STATES = [0, 1, 2, 3]

COMMON_PARAMS = {
    "load_knn_roberta": {
        "roberta_path": "../../data/roberta-QQP:v0"
    },
    "finetune_knn_roberta": {
        "top_k": 10,
        "evaluate_on_train": False,
        "training": {
            "max_steps": None,
            "evaluation_strategy": "steps",
            "eval_steps": 100,
            "save_steps": 100,
            "load_best_model_at_end": False,
            "per_device_train_batch_size": BATCH_SIZE,
            "per_device_eval_batch_size": 256,
        },
        "use_fakes": False,
        "add_intent": False
    },
    "evaluate_knn_roberta": {
        "top_k": 10,
        "batch_size": 256,
        "fake_similar": False,
        "prefix": None
    }
}

ARTIFACTS = {
    "dataset": "SOAD:v2",
    "roberta": "roberta-QQP:v0"
}

WANDB_ARGS = {
    "project": "aslan",
    "entity": "broccoliman",
    "job_type": "evaluation",
    "tags": ["no-seen", "debug", "knn-roberta", "series"]
}

MODULES = [
    ("encode_fshandler_labels", encode_fshandler_labels),
    ("load_knn_roberta", load_knn_roberta),
    ("finetune_knn_roberta", finetune_knn_roberta),
    ("evaluate_knn_roberta", evaluate_knn_roberta)
]

for support_size in POSSIBLE_SIZES:
    print(f"SUPPORT SIZE {support_size}")

    max_steps = min(int(5 * 150 * (support_size ** 2) * 2 / BATCH_SIZE + 1), 3000)
    COMMON_PARAMS["finetune_knn_roberta"]["training"]["max_steps"] = max_steps

    lab = FewShotLaboratory(
        modules=MODULES,
        pretraining_modules=[],
        artifacts=ARTIFACTS,
        support_size=support_size,
        extra_size=0,
        val_size=5,
        logger=print,
        wandb_args=WANDB_ARGS,
        params=COMMON_PARAMS,
        root_path="../../data")

    lab.init_data("SOAD:v2/CLINC150", -1)

    for random_state in RANDOM_STATES:
        print(f"RANDOM STATE {random_state}")

        lab.params["evaluate_knn_roberta"]["prefix"] = f"SS_{support_size}_RS_{random_state}"
        metrics, fshandler = lab.run(42)
