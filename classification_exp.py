import wandb
import os
import sys
import argparse
import json
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, \
    Trainer, TrainingArguments
from datasets import Dataset, load_metric, concatenate_datasets

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
        "--fake_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )

    return parser.parse_args()


def get_fakes(path, state):
    with open(path + f"/{state}-fakes.json") as f:
        fakes = json.load(f)

    good_intents = []
    good_texts = []

    for fake in fakes:
        good_intents.append(fake["intent"])
        good_texts.append(fake["text"])

    fake_dataset = Dataset.from_dict({"intent": good_intents, "text": good_texts})
    return fake_dataset


def main():
    args = parse_args()

    os.environ["OFFLINE"] = "True"
    os.environ["WANDB_MODE"] = "offline"
    sys.path.append("narnia")

    from pipeline import FewShotLaboratory, encode_fshandler_labels
    from environment import TokenizedDataset

    wandb.login()

    lab = FewShotLaboratory(
        modules=[("encode_fshandler_labels", encode_fshandler_labels)],
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

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train = TokenizedDataset(fshandler.known, lambda x: x["text"], tokenizer)
    test = TokenizedDataset(fshandler.unknown, lambda x: x["text"], tokenizer)

    metric = load_metric("../portal/metrics/accuracy.py")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    config = {
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "max_steps": 1000,
        "learning_rate": 5e-5,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "seed": 42,
        "lr_scheduler_type": "linear",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "gradient_accumulation_steps": 1
    }

    run = wandb.init(project="aslan",
                    tags=["hpc", "no-fakes", f"state-{args.state}", args.model],
                    job_type="training")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        eval_steps=20,
        logging_dir='./logs',
        logging_steps=10,
        logging_strategy="steps",
        report_to="wandb",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        disable_tqdm=False,
        **config
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=150)

    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    )
    trainer.train()

    final_metrics = trainer.evaluate(train, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(test, metric_key_prefix="test"))

    wandb.log(final_metrics)
    run.finish()

    fake_dataset = get_fakes(args.fake_path, args.state)

    top = fake_dataset.map(lambda x: {"label": fshandler.state["classlabel"].str2int(x["intent"])}, 
                           load_from_cache_file=False)

    train = TokenizedDataset(concatenate_datasets([fshandler.known, top]), lambda x: x["text"], tokenizer)
    test = TokenizedDataset(fshandler.unknown, lambda x: x["text"], tokenizer)

    run = wandb.init(project="aslan",
                    tags=["hpc", "with-fakes", f"state-{args.state}", args.model],
                    job_type="training")

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="no",
        eval_steps=20,
        logging_dir='./logs',
        logging_steps=10,
        logging_strategy="steps",
        report_to="wandb",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        disable_tqdm=False,
        **config
    )

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=150)

    trainer = Trainer(
        model=model,
        args=training_args, 
        train_dataset=train,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    )
    trainer.train()

    final_metrics = trainer.evaluate(train, metric_key_prefix="train")
    final_metrics.update(trainer.evaluate(test, metric_key_prefix="test"))

    wandb.log(final_metrics)
    run.finish()


if __name__ == "__main__":
    main()