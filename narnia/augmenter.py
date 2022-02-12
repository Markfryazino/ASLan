import torch
import wandb

import numpy as np

from bert_score import score
from datasets import ClassLabel, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
                         DataCollatorWithPadding
from tqdm.auto import tqdm, trange

from few_shot_training import naive_finetuning
from environment import TokenizedDataset


class Augmenter:
    def __init__(self, known, val_known, unknown):
        self.known = known
        self.val_known = val_known
        self.unknown = unknown

        self.classlabel = None
        self.init_classlabel()
        self.unique_intents = self.known.unique("intent")

        self.fake_dataset = None
        self.corrector_model = None
        self.corrector_trainer = None
        self.corrector_tokenizer = None

    def init_classlabel(self):
        unique_intents = list(self.known.unique("intent"))
        self.classlabel = ClassLabel(names=unique_intents)
        self.known = self.known.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                    load_from_cache_file=False)
        self.val_known = self.val_known.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                            load_from_cache_file=False)
        self.unknown = self.unknown.map(lambda x: {"label": self.classlabel.str2int(x["intent"]), **x},
                                        load_from_cache_file=False)

    def init_fakes_from_wandb(self, path):
        api = wandb.Api()
        run = api.run(path)
        f = run.file("results/fakes.json").download(replace=True)

        with open("results/fakes.json") as f:
            fakes = json.load(f)

        texts = []
        intents = []

        for fake in fakes:
            texts.append(fake["fake_text"])
            intents.append(fake["intent"])

        self.fake_dataset = Dataset.from_dict({"intent": intents, "text": texts})

    def train_generator(self, params):
        pass

    def generate(self, params):
        pass

    def train_corrector(self, params):
        tokenizer = AutoTokenizer.from_pretrained(params["model"])
        model = AutoModelForSequenceClassification.from_pretrained(params["model"])

        training_params = None if "training" not in params else params["training"]
        default_wandb_args = {
            "project": "aslan",
            "entity": "broccoliman",
            "job_type": "training",
            "tags": ["train_corrector"]
        }

        wandb_args = default_wandb_args if "wandb_args" not in params else params["wandb_args"]

        self.corrector_tokenizer = tokenizer
        self.corrector_model, self.corrector_trainer = naive_finetuning(model, tokenizer, self.known, 
                self.val_known, self.unknown, wandb_args, params["training"])

    def correct(self, params):
        def novel_filtering(x):
            return x["true_proba"] > params["threshold"]

        def _extract_thoughts(x, idx):
            top_n = 5

            ints, probs = [], []
            for i in range(top_n):
                intent = self.classlabel.int2str(argsorted_probas[idx, i].item())
                probs.append(sorted_probas[idx, i].item())
                ints.append(intent)

            true_intent_label = self.classlabel.str2int(x["intent"])
            true_proba = probas[idx, true_intent_label].item()

            return {"pred_intents": ints, "pred_probs": probs, "true_proba": true_proba}

        fakes_for_correcting = self.fake_dataset.remove_columns(["intent"])
        fakes_tokenized = TokenizedDataset(fakes_for_correcting, lambda x: x["text"], self.correcter_tokenizer)

        predictions = self.corrector_trainer.predict(fakes_tokenized)
        probas = torch.nn.Softmax(dim=1)(torch.tensor(predictions[0])).numpy()
        sorted_probas = np.sort(probas, axis=1)[:,::-1]
        argsorted_probas = np.argsort(probas, axis=1)[:,::-1]

        revealed = self.fake_dataset.map(extract_thoughts, with_indices=True, load_from_cache_file=False)
        self.filtered_fakes = revealed.filter(novel_filtering, load_from_cache_file=False)

    def diversify(self, params):
        self.aug_fakes = Dataset.from_dict({"intent": [], "text": []})

        if "model_type" not in params:
            params["model_type"] = "roberta-base"

        for intent in tqdm(self.unique_intents):
            good_texts = self.filtered_fakes.filter(lambda x: x["intent"] == intent,
                                                    load_from_cache_file=False)["text"]

            real_texts = self.known.filter(lambda x: x["intent"] == intent, load_from_cache_file=False)["text"]

            left_good_texts = np.array(good_texts).repeat(len(good_texts)).tolist()
            right_good_texts = good_texts * len(good_texts)

            left_texts_for_real = np.array(good_texts).repeat(len(real_texts)).tolist()
            right_texts_for_real = real_texts * len(good_texts)

            P, R, F = score(left_good_texts, right_good_texts, model_type=params["model_type"]) 
            P_real, R_real, F_real = score(left_texts_for_real, right_texts_for_real, model_type=params["model_type"])

            R = R.reshape((len(good_texts), len(good_texts)))
            R_real = R_real.reshape((len(good_texts), len(real_texts)))
            R = R.numpy()
            np.fill_diagonal(R, 0)
            R_real = R_real.numpy()

        bad_idxs = []

        while len(bad_idxs) + params["intent_size"] < len(good_texts):
            R_argmax = R.argmax()
            R_real_argmax = R_real.argmax()

            i = R_argmax // R.shape[1]
            j = R_argmax % R.shape[1]

            i_real = R_real_argmax // R_real.shape[1]
            j_real = R_real_argmax % R_real.shape[1]

            bad_i = i if R[i, j] > R_real[i_real, j_real] else i_real
            bad_idxs.append(bad_i)

            R[bad_i,:] = 0
            R[:,bad_i] = 0
            R_real[bad_i,:] = 0

        good_idxs = set(range(len(good_texts))) - set(bad_idxs)
        final_texts = [good_texts[i] for i in good_idxs]
        tmp_dataset = Dataset.from_dict({"text": final_texts, "intent": [intent] * params["intent_size"]})
        self.aug_fakes = concatenate_datasets([self.aug_fakes, tmp_dataset])
