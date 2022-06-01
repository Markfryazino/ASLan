import torch
import wandb
import json

import numpy as np

from pathlib import Path
from bert_score import score, BERTScorer
from datasets import ClassLabel, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
                         DataCollatorWithPadding, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from tqdm.auto import tqdm, trange
from collections import defaultdict
from inspect import getfullargspec
from sklearn.cluster import SpectralClustering
from scipy.special import softmax

from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate

from few_shot_training import naive_finetuning
from environment import TokenizedDataset, T5TokenizedDataset, SBERTDataset


def evaluate_prompt_model(prompt_model, dataloader):
    prompt_model.eval()
    total_loss = total_num = 0
    with torch.inference_mode():
        for inputs in tqdm(dataloader):
            loss = prompt_model(inputs.cuda())
            total_loss += loss.item()
            total_num += inputs["input_ids"].size(0)
    return total_loss / total_num


class Augmenter:
    def __init__(self, known, state, val_known=None, unknown=None, results_path="results"):
        self.known = known
        self.val_known = val_known
        self.unknown = unknown

        self.results = results_path
        Path(self.results).mkdir(parents=True, exist_ok=True)
        self.state = state

        self.classlabel = None
        self.init_classlabel()
        self.unique_intents = self.known.unique("intent")
        self.support_size = len(self.known) // len(self.unique_intents)

        self.fake_dataset = None
        self.corrector_model = None
        self.corrector_trainer = None
        self.corrector_tokenizer = None
        self.filtered_fakes = None
        self.aug_fakes = None

        self.condition_dataloader = None
        self.raw_condition = None
        self.prompt_model = None

        self.info = {}
        self.logs = {}

        self.config = {}

    def init_classlabel(self):
        unique_intents = list(self.known.unique("intent"))
        self.classlabel = ClassLabel(names=unique_intents)
        self.known = self.known.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                    load_from_cache_file=False)

        if self.val_known is not None:
            self.val_known = self.val_known.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                                load_from_cache_file=False)

        if self.unknown is not None:
            self.unknown = self.unknown.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                            load_from_cache_file=False)

    def init_fakes_from_wandb(self, path):
        api = wandb.Api()
        run = api.run(path)
        f = run.file(f"{self.results}/{self.state}-fakes.json").download(replace=True)

        with open(f"{self.results}/{self.state}-fakes.json") as f:
            fakes = json.load(f)

        texts = []
        intents = []

        for fake in fakes:
            texts.append(fake["fake_text"])
            intents.append(fake["intent"])

        self.fake_dataset = Dataset.from_dict({"intent": intents, "text": texts})

    def train_generator(self, num_easy_positives=None, sbert_path="all-mpnet-base-v2", t5_path="t5-base",
                        prefix_length=30, batch_size=4, max_length=64, num_train_epochs=3,
                        log_steps=25, grad_accum=8, lr=5e-5, wandb_args=None):

        default_wandb_args = {
            "project": "aslan",
            "entity": "broccoliman",
            "job_type": "training",
            "tags": ["prefix-tuning", "augmentation"],
            "save_code": True
        }

        wandb_args = wandb_args or default_wandb_args
        num_easy_positives = num_easy_positives or self.support_size // 2

        self.config["train_generator"] = {}
        for param in getfullargspec(self.train_generator)[0]:
            if param not in ["self", "wandb_args"]:
                self.config["train_generator"][param] = eval(param)

        positive_params = {
            "pair_numbers": {
                "hard_positive": 0,
                "hard_negative": 0,
                "easy_positive": num_easy_positives,
                "easy_negative": 0
            }
        }
        condition_params = {
            "pair_numbers": {
                "hard_positive": 0,
                "hard_negative": 0,
                "easy_positive": 1,
                "easy_negative": 0
            }
        }

        raw_train = SBERTDataset(self.known, **positive_params, sbert=sbert_path)

        raw_test = None if self.val_known is None else \
            SBERTDataset(self.val_known, **positive_params, sbert=sbert_path)
        self.raw_condition = SBERTDataset(self.known, **condition_params, sbert=sbert_path)

        dataset = {}
        dataset["train"] = []
        for i, val in enumerate(raw_train):
            dataset["train"].append(InputExample(
                guid=i,
                tgt_text=val["other_text"],
                text_a=val["source_text"],
                text_b=val["source_intent"],
            ))

        dataset["validation"] = None
        if raw_test is not None:
            dataset["validation"] = []
            for i, val in enumerate(raw_test):
                dataset["validation"].append(InputExample(
                    guid=i,
                    tgt_text=val["other_text"],
                    text_a=val["source_text"],
                    text_b=val["source_intent"],
                ))

        dataset["condition"] = []
        for i, val in enumerate(self.raw_condition):
            dataset["condition"].append(InputExample(
                guid=i,
                tgt_text=val["other_text"],
                text_a=val["source_text"],
                text_b=val["source_intent"],
            ))

        plm, tokenizer, model_config, WrapperClass = load_plm("t5", t5_path)
        template_text = ' {"placeholder":"text_b"} {"placeholder":"text_a"} {"special": "<eos>"} {"mask"} '
        mytemplate = PrefixTuningTemplate(model=plm, tokenizer=tokenizer, num_token=prefix_length,
                                          text=template_text)
    
        train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length, decoder_max_length=max_length, 
            batch_size=batch_size, shuffle=True, teacher_forcing=True, predict_eos_token=True,
            truncate_method="head")

        validation_dataloader = None
        if dataset["validation"] is not None:
            validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, 
                tokenizer=tokenizer, tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length, 
                decoder_max_length=max_length, batch_size=batch_size, shuffle=True, teacher_forcing=True, 
                predict_eos_token=True, truncate_method="head")

        self.condition_dataloader = PromptDataLoader(dataset=dataset["condition"], template=mytemplate, tokenizer=tokenizer, 
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length, decoder_max_length=max_length, 
            batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=True,
            truncate_method="head")

        prompt_model = PromptForGeneration(plm=plm, template=mytemplate, freeze_plm=True, tokenizer=tokenizer, 
                                           plm_eval_mode=True)
        prompt_model = prompt_model.cuda()

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) \
                    and p.requires_grad],
                "weight_decay": 0.1,
            },
            {
                "params": [p for n, p in mytemplate.named_parameters() if any(nd in n for nd in no_decay) \
                    and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8)
        total_steps  = len(train_dataloader) * num_train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps // grad_accum)

        wandb.init(**wandb_args)

        global_step = 0 
        total_loss = 0
        log_num = 0

        with tqdm(total=len(train_dataloader) * num_train_epochs) as pbar:
            for epoch in range(num_train_epochs):
                prompt_model.train()
                for step, inputs in enumerate(train_dataloader):
                    global_step += 1
                    loss = prompt_model(inputs.cuda())
                    loss.backward()
                    total_loss += loss.item()

                    log_num += inputs["input_ids"].size(0)
                    torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)

                    if global_step % grad_accum == 0:
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    if global_step % log_steps == 0: 
                        metrics = {
                            "epoch": epoch,
                            "global_step": global_step,
                            "train_loss": total_loss / log_steps,
                            "learning_rate": scheduler.get_last_lr()[0]
                        }
                        wandb.log(metrics, step=global_step)
                        print(metrics)
                        total_loss = 0
                        log_num = 0

                    pbar.update(1)

                if validation_dataloader is not None:
                    val_loss = evaluate_prompt_model(prompt_model, validation_dataloader)
                    print(f"Epoch {epoch}, validation loss: {val_loss}")
                    wandb.log({"epoch": epoch, "val_loss": val_loss, "global_step": global_step}, step=global_step)

        self.prompt_model = prompt_model
        
        wandb.finish()

    def generate(self, generation_args=None, multiplier=20, wandb_args=None):
        default_generation_args = {
            "max_length": 64,
            "temperature": .9,
            "do_sample": True,
            "top_k": 80,
            "top_p": 0.995,
            "repetition_penalty": 1.0,
        }

        generation_args = generation_args or default_generation_args

        self.config["generate"] = {}
        for param in getfullargspec(self.generate)[0]:
            if param not in ["self", "wandb_args"]:
                self.config["generate"][param] = eval(param)

        default_wandb_args = {
            "project": "aslan",
            "entity": "broccoliman",
            "job_type": "training",
            "tags": ["prefix-tuning", "augmentation"],
            "save_code": True
        }

        wandb_args = wandb_args or default_wandb_args

        wandb.init(**wandb_args)

        fakes = []
        steps = 0
        wandb.log({"total_steps": multiplier * len(self.condition_dataloader)})
        with torch.inference_mode():
            with tqdm(total=multiplier * len(self.condition_dataloader)) as pbar:
                for i in range(multiplier):
                    for inputs in self.condition_dataloader:
                        try:
                            fake_texts = self.prompt_model.generate(inputs.cuda(), **generation_args)[1]

                            for j, fake in enumerate(fake_texts):
                                raw = self.raw_condition[inputs["guid"][j].item()]
                                intent = raw["source_intent"]
                                source_text = raw["source_text"]

                                fakes.append({
                                    "intent": intent,
                                    "source_text": source_text,
                                    "fake_text": fake
                                })

                            pbar.update(1)
                            wandb.log({"step": steps})
                            steps += 1
                        except:
                            print("беда...")

        with open(f"{self.results}/{self.state}-fakes.json", "w") as f:
            json.dump(fakes, f)

        with open(f"{self.results}/{self.state}-comparison.txt", "w") as f:
            for intent in self.known.unique("intent"):
                f.write(f"\n\n\n\nINTENT: {intent}\n\n\nREAL ANCHOR\n\n")
                for row in self.known:
                    if row["intent"] != intent:
                        continue
                    f.write(f"{row['text']}\n")

                f.write(f"\n\nFAKE\n\n")
                for fake in fakes:
                    if fake["intent"] != intent:
                        continue
                    f.write(f"{fake['fake_text']}\n")

        wandb.save(f"{self.results}/{self.state}-fakes.json")
        wandb.save(f"{self.results}/{self.state}-comparison.txt")
        wandb.finish()

        texts = []
        intents = []

        for row in fakes:
            texts.append(row["fake_text"])
            intents.append(row["intent"])

        self.fake_dataset = Dataset.from_dict({"text": texts, "intent": intents})

    def train_corrector(self, model="roberta-base", training_args=None, wandb_args=None):
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model, num_labels=len(self.unique_intents))

        default_wandb_args = {
            "project": "aslan",
            "entity": "broccoliman",
            "job_type": "training",
            "tags": ["train_corrector"]
        }

        self.config["train_corrector"] = {}
        for param in getfullargspec(self.train_corrector)[0]:
            if param not in ["self", "wandb_args"]:
                self.config["train_corrector"][param] = eval(param)

        wandb_args = wandb_args or default_wandb_args

        self.corrector_tokenizer = tokenizer
        self.corrector_model, self.corrector_trainer = naive_finetuning(model, tokenizer, self.known, 
                self.val_known, self.unknown, wandb_args, training_args)

    def correct(self, threshold):
        def novel_filtering(x):
            return x["true_proba"] > threshold

        def extract_thoughts(x, idx):
            top_n = 5

            ints, probs = [], []
            for i in range(top_n):
                intent = self.classlabel.int2str(argsorted_probas[idx, i].item())
                probs.append(sorted_probas[idx, i].item())
                ints.append(intent)

            true_intent_label = self.classlabel.str2int(x["intent"])
            true_proba = probas[idx, true_intent_label].item()

            return {"pred_intents": ints, "pred_probs": probs, "true_proba": true_proba}

        self.config["correct"] = {}
        for param in getfullargspec(self.correct)[0]:
            if param not in ["self", "wandb_args"]:
                self.config["correct"][param] = eval(param)

        fakes_for_correcting = self.fake_dataset.remove_columns(["intent"])
        fakes_tokenized = TokenizedDataset(fakes_for_correcting, lambda x: x["text"], self.corrector_tokenizer)

        predictions = self.corrector_trainer.predict(fakes_tokenized)
        probas = torch.nn.Softmax(dim=1)(torch.tensor(predictions[0])).numpy()
        sorted_probas = np.sort(probas, axis=1)[:,::-1]
        argsorted_probas = np.argsort(probas, axis=1)[:,::-1]

        revealed = self.fake_dataset.map(extract_thoughts, with_indices=True, load_from_cache_file=False)
        self.info["corrector_revealed"] = revealed

        true_prob_array = np.array(self.info["corrector_revealed"]["true_proba"])
        self.logs["true_prob_median"] = np.median(true_prob_array)
        self.logs["true_prob_mean"] = true_prob_array.mean()
        self.logs["true_prob_filtered_ratio"] = (true_prob_array > threshold).sum() / true_prob_array.shape[0]
        self.logs["true_prob_filtered_mean"] = true_prob_array[true_prob_array > threshold].mean()
        self.logs["true_prob_filtered_median"] = np.median(true_prob_array[true_prob_array > threshold])

        self.filtered_fakes = revealed.filter(novel_filtering, load_from_cache_file=False)

    def diversify(self, intent_size, temperature=0.05, model_type="roberta-base", num_layers=10, wandb_args=None):

        self.config["diversify"] = {}
        for param in getfullargspec(self.diversify)[0]:
            if param not in ["self", "wandb_args"]:
                self.config["diversify"][param] = eval(param)

        default_wandb_args = {
            "project": "aslan",
            "entity": "broccoliman",
            "job_type": "training",
            "tags": ["diversify"]
        }

        wandb_args = wandb_args or default_wandb_args
        wandb.init(**wandb_args)

        self.logs["diversify"] = {}

        aug_fakes_lists = []
        
        scorer = BERTScorer(model_type=model_type, num_layers=num_layers)

        good_texts_by_intent = defaultdict(lambda: [])
        real_texts_by_intent = defaultdict(lambda: [])

        for pair in self.filtered_fakes:
            good_texts_by_intent[pair["intent"]].append(pair["text"])

        for pair in self.known:
            real_texts_by_intent[pair["intent"]].append(pair["text"])

        self.info["diversify"] = {}

        for intent in tqdm(self.unique_intents):
            good_texts = good_texts_by_intent[intent]
            real_texts = real_texts_by_intent[intent]

            left_good_texts = np.array(good_texts).repeat(len(good_texts)).tolist()
            right_good_texts = good_texts * len(good_texts)

            left_texts_for_real = np.array(good_texts).repeat(len(real_texts)).tolist()
            right_texts_for_real = real_texts * len(good_texts)

            P, R, F = scorer.score(left_good_texts, right_good_texts) 
            P_real, R_real, F_real = scorer.score(left_texts_for_real, right_texts_for_real)

            R = R.reshape((len(good_texts), len(good_texts)))
            R_real = R_real.reshape((len(good_texts), len(real_texts)))
            R = R.numpy()
            R_real = R_real.numpy()

            self.info["diversify"][intent] = {
                "R": R.copy(),
                "R_real": R_real.copy()
            }

            if R.shape[0] < intent_size:
                print(f"{intent_size} is less than {R.shape[0]}")

            true_intent_size = min(intent_size, R.shape[0])

            y = SpectralClustering(assign_labels='kmeans', affinity="precomputed", 
                                   n_clusters=true_intent_size).fit_predict(R)
            real_dists = R_real.max(axis=1)

            good_idxs = []

            for cl in range(true_intent_size):
                idxs = np.where(y == cl)[0]
                p = softmax(real_dists[idxs] / temperature)
                idx = np.random.choice(idxs, p=p)
                good_idxs.append(idx)

            self.info["diversify"][intent]["good_idxs"] = good_idxs

            R_cp = self.info["diversify"][intent]["R"]
            R_real_cp = self.info["diversify"][intent]["R_real"]
            np.fill_diagonal(R_cp, 0)

            self.logs["diversify"][intent] = {
                "R_max_mean": R_cp.max(axis=1).mean(),
                "R_good_max_mean": R_cp[good_idxs, :][:, good_idxs].max(axis=1).mean(),
                "R_real_max_mean": R_real_cp.max(axis=1).mean(),
                "R_real_good_max_mean": R_real_cp[good_idxs, :].max(axis=1).mean(),
                "R_max_median": np.median(R_cp.max(axis=1)),
                "R_good_max_median": np.median(R_cp[good_idxs, :][:, good_idxs].max(axis=1)),
                "R_real_max_median": np.median(R_real_cp.max(axis=1)),
                "R_real_good_max_median": np.median(R_real_cp[good_idxs, :].max(axis=1)),              
            }

            final_texts = [good_texts[i] for i in good_idxs]
            tmp_dataset = Dataset.from_dict({"text": final_texts, "intent": [intent] * len(final_texts)})
            aug_fakes_lists.append(tmp_dataset)
        
        self.logs.update({
            "R_good_max_mean": np.mean([self.logs["diversify"][intent]["R_good_max_mean"] for \
                intent in self.unique_intents]),
            "R_max_mean": np.mean([self.logs["diversify"][intent]["R_max_mean"] for \
                intent in self.unique_intents]),
            "R_real_good_max_mean": np.mean([self.logs["diversify"][intent]["R_real_good_max_mean"] for \
                intent in self.unique_intents]),
            "R_real_max_mean": np.mean([self.logs["diversify"][intent]["R_real_max_mean"] for \
                intent in self.unique_intents])    
        })

        self.aug_fakes = concatenate_datasets(aug_fakes_lists)

        fakes = []
        for element in self.aug_fakes:
            fakes.append(element)

        with open(f"{self.results}/{self.state}-fakes.json", "w") as f:
            json.dump(fakes, f)

        with open(f"{self.results}/{self.state}-comparison.txt", "w") as f:
            for intent in self.unique_intents:
                f.write(f"\n\n\n\nINTENT: {intent}\n\n\nREAL ANCHOR\n\n")
                for row in self.known:
                    if row["intent"] != intent:
                        continue
                    f.write(f"{row['text']}\n")

                f.write(f"\n\nFAKE\n\n")
                for fake in fakes:
                    if fake["intent"] != intent:
                        continue
                    f.write(f"{fake['text']}\n")

        wandb.save(f"{self.results}/{self.state}-fakes.json")
        wandb.save(f"{self.results}/{self.state}-comparison.txt")

        checked_texts = set()
        chosen_texts = set(self.aug_fakes["text"])
        probs = []
        for rev in self.info["corrector_revealed"]:
            if rev["text"] in chosen_texts and rev["text"] not in checked_texts:
                checked_texts.add(rev["text"])
                probs.append(rev["true_proba"])

        self.logs.update({
            "true_prob_good_mean": np.mean(probs),
            "true_prob_good_median": np.median(probs)
        })

        self.logs.update({
            "3metric": np.mean([self.logs["R_max_mean"] / 2, self.logs["R_real_max_mean"] / 2, \
                self.logs["true_prob_good_mean"]])
        })

        wandb.log(self.logs)
        wandb.finish()
