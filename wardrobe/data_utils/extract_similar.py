import wandb
import numpy as np
import torch
import json
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding, RobertaTokenizerFast, \
                         RobertaForSequenceClassification
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy
from tqdm.notebook import trange


splits = []
for part in ["train", "val", "test"]:
    splits.append(f"seen_{part}")

raw = load_dataset("csv", data_files={el: f"artifacts/CLINC150:v4/zero_shot_split/{el}.csv" for el in splits})
dataset = raw.filter(lambda x: x["intent"] != "oos")

unique_intents = dataset["seen_train"].unique("intent")
classlabel = ClassLabel(names=unique_intents)


def encode_label(x):
    x["label"] = classlabel.str2int(x["intent"])
    return x


dataset = dataset.map(encode_label)
​
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2').cuda()
​
def embed(x):   
    x["embedding"] = embedder.encode(x["text"])
    return x

dataset = dataset.map(embed, batched=True)

intent_embeddings = embedder.encode(classlabel.names)

def get_top_k_for_each_intent(k):
    intent_sims = scipy.spatial.distance.pdist(intent_embeddings, metric="cosine")
    intent_sims = scipy.spatial.distance.squareform(intent_sims)
    intent_sims[np.where(intent_sims == 0)] = 3
    argsorted = intent_sims.argsort(axis=1)
    res = {}
    for i, name in enumerate(classlabel.names):
        cur = []
        for j in range(k):
            cur.append((classlabel.names[argsorted[i, j]], intent_sims[i, argsorted[i, j]]))
        res[name] = cur
    return res

intents_similar_20 = get_top_k_for_each_intent(20)

def get_top_k_for_each_row(k):
    same_intents = scipy.spatial.distance.pdist(np.array(dataset["seen_train"]["label"]).reshape(-1, 1), metric=lambda x, y: int(x[0] != y[0]))

    row_embed = scipy.spatial.distance.pdist(dataset["seen_train"]["embedding"], metric="cosine")
    row_embed = scipy.spatial.distance.squareform(row_embed * same_intents)
    
    row_embed[np.where(row_embed == 0.)] = 3

    argsorted = row_embed.argsort(axis=1)
    res = []
    for i in trange(len(dataset["seen_train"])):
        cur = []
        for j in range(k):
            cur.append((int(argsorted[i, j]), row_embed[i, argsorted[i, j]]))
        res.append(cur)
    return res

rows_similar_100 = get_top_k_for_each_row(100)

run = wandb.init(project="aslan", job_type="data-preprocessing",
                 notes="Extract the most similar utterances for seen_train")
run.use_artifact("CLINC150:v4")

!mkdir similar_data

with open("similar_data/intents.json", "w") as f:
    json.dump(intents_similar_20, f)
    
with open("similar_data/utterances.json", "w") as f:
    json.dump(rows_similar_100, f)
    
my_data = wandb.Artifact("CLINC150-seen-similar", type="dataset", description="Indices of most similar intents and utterances in CLINC150-seen")

my_data.add_dir("similar_data")
run.log_artifact(my_data)
    
run.finish()
