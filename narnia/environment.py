import wandb
import numpy as np
import scipy
import scipy.spatial.distance
import torch
import json
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets, Dataset
from transformers import DataCollatorWithPadding
from sentence_transformers import SentenceTransformer, InputExample
import os
import pandas as pd
from tqdm.auto import tqdm, trange
from datasets import set_caching_enabled


set_caching_enabled(False)


def load_split_dataset(root_path, dataset, split):
    mapping = {el: os.path.join(root_path, dataset, el + ".csv") for el in ["train", "valid", "test"]}
    mapping["val"] = mapping["valid"]
    del mapping["valid"]

    raw = load_dataset("csv", data_files=mapping)
    with open(os.path.join(root_path, dataset, "splits.txt")) as f:
        buckets = eval(f.read())

    if split == -1:
        unseen = []
        for bucket in buckets:
            unseen += bucket
    else:
        unseen = buckets[split]

    seen_data = raw.filter(lambda x: x["intent"] not in unseen, load_from_cache_file=False)
    unseen_data = raw.filter(lambda x: x["intent"] in unseen, load_from_cache_file=False)
    return seen_data, concatenate_datasets([unseen_data["train"], unseen_data["val"], unseen_data["test"]])


def load_from_memory(root_path="artifacts/CLINC150:v5"):
    splits = []
    for part in ["train", "val", "test"]:
        for see in ["seen", "unseen"]:
            splits.append(f"{see}_{part}")

    raw = load_dataset("csv", data_files={el: os.path.join(root_path, el + ".csv") for el in splits})
    raw = raw.filter(lambda x: x["intent"] != "oos", load_from_cache_file=False)
    raw_unseen = concatenate_datasets([raw["unseen_train"], raw["unseen_val"], raw["unseen_test"]])
    return raw, raw_unseen


def load_unseen(root_path="artifacts/CLINC150:v5"):
    raw = load_dataset("csv", data_files=os.path.join(root_path, "unseen.csv"))
    raw = raw.filter(lambda x: x["intent"] != "oos", load_from_cache_file=False)
    return raw["train"]


def set_generator(dataset, support_size=10, shuffle=True):
    unique_intents = dataset.unique("intent")
    full_intents = np.array(dataset["intent"])
    intent_idxs = {}

    for intent in unique_intents:
        intent_idxs[intent] = np.where(full_intents == intent)[0]

    while True:
        train, test = [], []
        for intent in unique_intents:
            train_ids = np.random.choice(intent_idxs[intent], support_size, replace=False)
            test_ids = list(set(intent_idxs[intent]) - set(train_ids))
            train.append(dataset.select(train_ids))
            test.append(dataset.select(test_ids))

        if shuffle:
            yield concatenate_datasets(train).shuffle(load_from_cache_file=False).flatten_indices(), \
                  concatenate_datasets(test).shuffle(load_from_cache_file=False).flatten_indices()
        else:
            yield concatenate_datasets(train).flatten_indices(), concatenate_datasets(test).flatten_indices()


def encode_example(example, mapping):
    example["label"] = mapping[example["intent"]]
    return example


class UIDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, intents):
        self.source = source_dataset
        self.intents = intents
        self.n = len(source_dataset)
        self.m = len(self.intents)
    
    def __len__(self):
        return self.n * self.m
    
    def __getitem__(self, idx):
        u_id = idx // self.m
        i_id = idx % self.m
        return {
            "text": self.source[u_id]["text"],
            "intent": self.intents[i_id],
            "label": int(self.source[u_id]["intent"] == self.intents[i_id])
        }
    

class UUDataset(torch.utils.data.Dataset):
    def __init__(self, known, unknown):
        self.known = known
        self.unknown = unknown
        self.n = len(unknown)
        self.m = len(known)
    
    def __len__(self):
        return self.n * self.m
    
    def __getitem__(self, idx):
        un_id = idx // self.m
        kn_id = idx % self.m
        return {
            "text_unknown": self.unknown[un_id]["text"],
            "text_known": self.known[kn_id]["text"],
            "label": int(self.unknown[un_id]["intent"] == self.known[kn_id]["intent"])
        }

class STUUDataset(torch.utils.data.Dataset):
    def __init__(self, known, unknown, logger=print, sbert=None, top_k=10, device="cuda"):
        self.known = known
        self.unknown = unknown
        self.top_k = top_k
        self.n = len(unknown)
        self.m = len(known)

        if sbert is None:
            logger("Parameter sbert is None, initializing as 'all-mpnet-base-v2'")
            sbert = SentenceTransformer('all-mpnet-base-v2').to(device)
        elif type(sbert) is str:
            sbert = SentenceTransformer(sbert).to(device)
    
        logger("Encoding known dataset using SBERT...")
        encoded_known = sbert.encode(self.known["text"])

        logger("Encoding unknown dataset using SBERT...")
        encoded_unknown = sbert.encode(self.unknown["text"])

        logger("Counting distance")
        un2kn = scipy.spatial.distance.cdist(encoded_unknown, encoded_known, metric="cosine")
        
        self.close_idxs = np.argpartition(un2kn, self.top_k)[:,:self.top_k]

    def __len__(self):
        return self.n * self.top_k
    
    def __getitem__(self, idx):
        un_id = idx // self.top_k
        kn_id = int(self.close_idxs[un_id, idx % self.top_k])
        return {
            "text_unknown": self.unknown[un_id]["text"],
            "text_known": self.known[kn_id]["text"],
            "label": int(self.unknown[un_id]["intent"] == self.known[kn_id]["intent"])
        }


class SBERTDataset(torch.utils.data.Dataset):
    def __init__(self, source_data, sbert=None, logger=print, pair_numbers=None, device="cuda"):
        self.pair_numbers = pair_numbers or {
            "hard_positive": 3,
            "hard_negative": 3,
            "easy_positive": 3,
            "easy_negative": 3
        }
        logger(f"Initializing SBERTDataset, pair_numbers: {self.pair_numbers}")

        self.source = source_data
        self.n = len(self.source)

        if sbert is None:
            logger("Parameter sbert is None, initializing as 'all-mpnet-base-v2'")
            sbert = SentenceTransformer('all-mpnet-base-v2').to(device)
        elif type(sbert) is str:
            sbert = SentenceTransformer(sbert).to(device)

        logger("Encoding dataset using SBERT...")
        embeddings = sbert.encode(self.source["text"])
        distances = scipy.spatial.distance.pdist(embeddings, metric="cosine")
        logger("Dataset was encoded")

        unique_intents = self.source.unique("intent")
        self.positive_texts = {}
        self.negative_texts = {}
        for intent in unique_intents:
            self.positive_texts[intent] = self.source.select(np.where(np.array(self.source["intent"]) == intent)[0])
            self.negative_texts[intent] = self.source.select(np.where(np.array(self.source["intent"]) != intent)[0])
        
        mask = scipy.spatial.distance.pdist(np.array(self.source["label"]).reshape((-1, 1)))
        negative_mask = (mask > 0).astype(int)
        positive_mask = (mask == 0).astype(int)

        negative_sform = scipy.spatial.distance.squareform(distances * negative_mask)
        negative_sform[np.where(negative_sform == 0.)] = 42
        self.hard_negatives = np.argpartition(negative_sform, 
                                              self.pair_numbers["hard_negative"])[:,:self.pair_numbers["hard_negative"]]

        del negative_mask, negative_sform
        logger("Negative indices were built!")

        positive_sform = scipy.spatial.distance.squareform(distances * positive_mask)
        positive_sform[np.where(positive_sform == 0.)] = -42
        self.hard_positives = np.argpartition(positive_sform, 
                                              -self.pair_numbers["hard_positive"])[:,-self.pair_numbers["hard_positive"]:]

        del positive_mask, positive_sform
        logger("Positive indices were built!")

    def __len__(self):
        return self.n * sum(self.pair_numbers.values())
    
    def __getitem__(self, idx):
        mode = idx % sum(self.pair_numbers.values())
        real_idx = idx // sum(self.pair_numbers.values())
        if mode < self.pair_numbers["hard_positive"]:
            other_idx = int(self.hard_positives[real_idx][mode])
            other_text = self.source[other_idx]["text"]
            other_intent = self.source[other_idx]["intent"]
            label = 1
        elif mode < self.pair_numbers["hard_positive"] + self.pair_numbers["hard_negative"]:
            other_idx = int(self.hard_negatives[real_idx][mode - self.pair_numbers["hard_positive"]])
            other_text = self.source[other_idx]["text"]
            other_intent = self.source[other_idx]["intent"]
            label = 0
        elif mode < self.pair_numbers["hard_positive"] + self.pair_numbers["hard_negative"] + self.pair_numbers["easy_positive"]:
            other_idx = np.random.choice(len(self.positive_texts[self.source[real_idx]["intent"]]))
            other_text = self.positive_texts[self.source[real_idx]["intent"]][other_idx]["text"]
            other_intent = self.positive_texts[self.source[real_idx]["intent"]][other_idx]["intent"]
            label = 1
        else:
            other_idx = np.random.choice(len(self.negative_texts[self.source[real_idx]["intent"]]))
            other_text = self.negative_texts[self.source[real_idx]["intent"]][other_idx]["text"]
            other_intent = self.negative_texts[self.source[real_idx]["intent"]][other_idx]["intent"]
            label = 0

        return {
            "source_text": self.source[real_idx]["text"],
            "source_intent": self.source[real_idx]["intent"],
            "other_text": other_text,
            "other_intent": other_intent,
            "label": label
        }


class IEFormatDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, samples=None):
        self.source = source_dataset
        self.idxs = np.arange(len(source_dataset))
        if samples is not None:
            self.idxs = np.random.choice(self.idxs, size=(samples), replace=False)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, idx):
        _idx = int(self.idxs[idx])
        return InputExample(texts=[self.source[_idx]["source_text"], self.source[_idx]["other_text"]],
                            label=float(self.source[_idx]["label"]))


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, builder, tokenizer, sample_size=None):
        self.source = source_dataset
        self.builder = builder
        self.tokenizer = tokenizer

        if type(sample_size) == float:
            sample_size = int(sample_size * len(self.source))
        self.sample_size = sample_size if sample_size is not None else len(self.source)
        self.use_idxs = np.arange(len(self.source))
        if self.sample_size < len(self.source):
            self.use_idxs = np.random.choice(len(self.source), self.sample_size, replace=False)
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        source_example = self.source[int(self.use_idxs[idx])]
        tokenized = self.tokenizer(self.builder(source_example))
        if "label" in source_example:
            tokenized.update({"label": source_example["label"]})
        return tokenized
    

class FewShotHandler():
    def __init__(self, unknown, known=None, device="cuda", logger=print):
        self.known = known
        self.unknown = unknown
        
        self.intents = self.unknown.unique("intent")
        self.intent_num = len(self.intents)
        self.intent2label = {intent: label for label, intent in enumerate(self.intents)}
        self.unknown = self.unknown.map(lambda x: encode_example(x, self.intent2label), batched=False,
                                        load_from_cache_file=False)

        if known is not None:
            self.known = self.known.map(lambda x: encode_example(x, self.intent2label), batched=False,
                                        load_from_cache_file=False)

            known_intents_array = np.array(known["intent"])
            self.known_intent_idxs = {}

            for intent in self.intents:
                self.known_intent_idxs[intent] = np.where(known_intents_array == intent)[0]
            
            self.uu_dataset = UUDataset(self.known, self.unknown)

        else:
            self.known_intent_idxs = None
            self.uu_dataset = None
            
        self.ui_dataset = UIDataset(self.unknown, self.intents)
        self.device = device

        self.stuu_dataset = None

        self.logger = logger

        self.state = {}

    def log(self, text):
        self.logger(text)

    def eval_ui(self, model, tokenizer, batch_size=64, separator="<sep>"):

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        tokenized_dataset = TokenizedDataset(self.ui_dataset, lambda x: x["text"] + separator + x["intent"], tokenizer)
        
        loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size,
                                             shuffle=False, collate_fn=collator)
        
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            predictions = []
            for batch in tqdm(loader):
                output = model(**batch.to(self.device))
                predictions.append(output["logits"])
                
        predictions = torch.cat(predictions)[:,1]
        
        correct = 0
        for idx in trange(0, len(self.ui_dataset), self.intent_num):
            current_prediction = torch.argmax(predictions[idx:idx + self.intent_num])
            correct += self.ui_dataset[idx + current_prediction.item()]["label"]
        return {"accuracy": correct / len(self.unknown)}
    
    def eval_as_1nn(self, model, tokenizer, dataset, batch_size, separator):

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        tokenized_dataset = TokenizedDataset(dataset, lambda x: x["text_unknown"] + \
                                             separator + x["text_known"], tokenizer)
        
        loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=collator)
        
        model.to(self.device)
        model.eval()

        with torch.no_grad():
            predictions = []
            for batch in tqdm(loader):
                output = model(**batch.to(self.device))
                predictions.append(output["logits"])
                
        predictions = torch.cat(predictions)[:,1]
        
        step = len(dataset) // len(self.unknown)
        correct = 0

        log_dataset = {
            "score": [],
            "winner": [],
        }
        for idx, data in enumerate(dataset):
            for key, val in data.items():
                if key not in log_dataset:
                    log_dataset[key] = []
                log_dataset[key].append(val)
                log_dataset["score"].append(predictions[idx].item())
                log_dataset["winner"].append(False)

        details = []
        for idx in trange(0, len(dataset), step):
            current_prediction = torch.argmax(predictions[idx:idx + step])
            correct += dataset[idx + current_prediction.item()]["label"]

            log_idx = idx + current_prediction.item()
            log_dataset["winner"][log_idx] = True
            current_details = {
                "text_unknown": dataset[log_idx]["text_unknown"],
                "text_known": dataset[log_idx]["text_known"],
                "label": dataset[log_idx]["label"],
                "prediction": predictions[log_idx].item(),
                "number_of_correct": sum([dataset[i]["label"] for i in range(idx, idx + step)])
            }
            details.append(current_details)

        self.state["eval_log_dataset"] = Dataset.from_dict(log_dataset)
        return {"accuracy": correct / len(self.unknown), "details": details}

    def eval_uu(self, model, tokenizer, batch_size=64, separator="<sep>"):
        return self.eval_as_1nn(model, tokenizer, self.uu_dataset, batch_size, separator)

    def eval_stuu(self, model, tokenizer, fake_known=None, sbert=None, top_k=10, batch_size=64, separator="<sep>"):
        if (sbert is None) and ("sbert" in self.state):
            self.log("Found sbert in fshandler state, using it")
            sbert = self.state["sbert"]

        self.log("Reinitializing STUU dataset")
        known_for_stuu = None
        if fake_known is not None:
            self.log("Using fake data for prediction!")
            fake_known = fake_known.map(lambda x: encode_example(x, self.intent2label), batched=False,
                                        load_from_cache_file=False)

            known_for_stuu = concatenate_datasets([self.known, fake_known])
        else:
            known_for_stuu = self.known

        self.stuu_dataset = STUUDataset(known_for_stuu, self.unknown, logger=self.logger, sbert=sbert, 
                                        top_k=top_k, device=self.device)
    
        return self.eval_as_1nn(model, tokenizer, self.stuu_dataset, batch_size, separator)

    def eval_pure_sbert(self, sbert=None):
        if (sbert is None) and ("sbert" in self.state):
            self.log("Found sbert in fshandler state, using it")
            sbert = self.state["sbert"]

        if (self.stuu_dataset is None) or (self.stuu_dataset.top_k != 1):
            self.log("Reinitializing STUU dataset")
            self.stuu_dataset = STUUDataset(self.known, self.unknown, logger=self.logger, sbert=sbert, 
                                            top_k=1, device=self.device)
        else:
            self.log("Using cached STUU dataset")
    
        acc = np.mean([row["label"] for row in self.stuu_dataset])
        return {"accuracy": acc}
