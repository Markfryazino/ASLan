import wandb
import numpy as np
import scipy
import torch
import json
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
from transformers import DataCollatorWithPadding
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
from tqdm.auto import tqdm, trange
import logging



def load_from_memory(root_path="artifacts/CLINC150:v4/zero_shot_split"):
    splits = []
    for part in ["train", "val", "test"]:
        for see in ["seen", "unseen"]:
            splits.append(f"{see}_{part}")

    raw = load_dataset("csv", data_files={el: os.path.join(root_path, el + ".csv") for el in splits})
    raw = raw.filter(lambda x: x["intent"] != "oos")
    raw_unseen = concatenate_datasets([raw["unseen_train"], raw["unseen_val"], raw["unseen_test"]])
    return raw, raw_unseen


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
            yield concatenate_datasets(train).shuffle().flatten_indices(), \
                  concatenate_datasets(test).shuffle().flatten_indices()
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
    def __init__(self, known, unknown, sbert, top_k=10, device="cuda"):
        self.known = known
        self.unknown = unknown
        self.top_k = top_k
        self.n = len(unknown)
        self.m = len(known)

        if sbert is None:
            logging.info("Parameter sbert is None, initializing as 'all-mpnet-base-v2'")
            sbert = SentenceTransformer('all-mpnet-base-v2').to(device)
        elif type(sbert) is str:
            sbert = SentenceTransformer(sbert).to(device)
    
        logging.debug("Encoding known dataset using SBERT...")
        encoded_known = sbert.encode(self.known["text"])

        logging.debug("Encoding unknown dataset using SBERT...")
        encoded_unknown = sbert.encode(self.unknown["text"])

        logging.debug("Counting distance")
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
    def __init__(self, unknown, known=None, device="cuda"):
        self.known = known
        self.unknown = unknown
        
        self.intents = self.unknown.unique("intent")
        self.intent_num = len(self.intents)
        self.intent2label = {intent: label for label, intent in enumerate(self.intents)}
        self.unknown = self.unknown.map(lambda x: encode_example(x, self.intent2label), batched=False)

        if known is not None:
            self.known = self.known.map(lambda x: encode_example(x, self.intent2label), batched=False)

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

        details = []
        for idx in trange(0, len(dataset), step):
            current_prediction = torch.argmax(predictions[idx:idx + step])
            correct += dataset[idx + current_prediction.item()]["label"]

            log_idx = idx + current_prediction.item()
            current_details = {
                "text_unknown": dataset[log_idx]["text_unknown"],
                "text_known": dataset[log_idx]["text_known"],
                "label": dataset[log_idx]["label"],
                "prediction": predictions[log_idx].item(),
                "number_of_correct": sum([dataset[i]["label"] for i in range(idx, idx + step)])
            }
            details.append(current_details)

        return {"accuracy": correct / len(self.unknown), "details": details}

    def eval_uu(self, model, tokenizer, batch_size=64, separator="<sep>"):
        return self.eval_as_1nn(model, tokenizer, self.uu_dataset, batch_size, separator)

    def eval_stuu(self, model, tokenizer, sbert=None, top_k=10, batch_size=64, separator="<sep>"):
        if (self.stuu_dataset is None) or (top_k != self.stuu_dataset.top_k):
            logging.debug("Reinitializing STUU dataset")
            self.stuu_dataset = STUUDataset(self.known, self.unknown, sbert, top_k, device=self.device)
        else:
            logging.debug("Using cached STUU dataset")
    
        return self.eval_as_1nn(model, tokenizer, self.stuu_dataset, batch_size, separator)
