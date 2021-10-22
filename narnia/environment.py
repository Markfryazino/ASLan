import wandb
import numpy as np
import torch
import json
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd
from tqdm.auto import tqdm, trange


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

    
class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, source_dataset, builder, tokenizer):
        self.source = source_dataset
        self.builder = builder
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        source_example = self.source[idx]
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
        self.unknown = self.unknown.map(lambda x: encode_example(x, self.intent2label), batched=True)

        if known is not None:
            self.known = self.known.map(lambda x: encode_example(x, self.intent2label), batched=True)

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

    def eval_ui(self, model, tokenizer, batch_size=64, separator="<sep>"):

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        tokenized_dataset = TokenizedDataset(self.ui_dataset, lambda x: x["text"] + separator + x["intent"], tokenizer)
        
        loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
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
    
    def eval_uu(self, model, tokenizer, batch_size=64, separator="<sep>"):

        collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
        tokenized_dataset = TokenizedDataset(self.uu_dataset, lambda x: x["text_unknown"] + \
                                             separator + x["text_known"], tokenizer)
        
        loader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
        
        model.to(self.device)
        model.eval()
        
        with torch.no_grad():
            predictions = []
            for batch in tqdm(loader):
                output = model(**batch.to(self.device))
                predictions.append(output["logits"])
                
        predictions = torch.cat(predictions)[:,1]
        
        correct = 0
        for idx in trange(0, len(self.uu_dataset), len(self.known)):
            current_prediction = torch.argmax(predictions[idx:idx + len(self.known)])
            correct += self.uu_dataset[idx + current_prediction.item()]["label"]
        return {"accuracy": correct / len(self.unknown)}
