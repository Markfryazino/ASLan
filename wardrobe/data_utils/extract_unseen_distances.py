#!g1.1

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


#!g1.1

splits = []
for part in ["train", "val", "test"]:
    splits.append(f"unseen_{part}")

raw = load_dataset("csv", data_files={el: f"artifacts/CLINC150:v4/zero_shot_split/{el}.csv" for el in splits})
dataset = raw.filter(lambda x: x["intent"] != "oos")

#!g1.1
dataset = concatenate_datasets([dataset["unseen_train"], dataset["unseen_val"], dataset["unseen_test"]])

#!g1.1

embedder = SentenceTransformer('all-mpnet-base-v2').cuda()

dataset = dataset.map(lambda x: {"embedding": embedder.encode(x["text"]), **x}, batched=True)

#!g1.1

from random import randint
import scipy

embeds = np.array(dataset["embedding"])

i = randint(0, len(dataset))

cur_embed = embeds[i]

sims = -scipy.spatial.distance.cdist(cur_embed.reshape(1, -1), embeds, metric="cosine")

ind = np.argpartition(sims[0], -5)[-5:]

for idx in ind:
    idx = int(idx)
    print(dataset[idx]["intent"])
    print(dataset[idx]["text"])
    print(sims[0, idx])
    print("\n")

#!g1.1

run = wandb.init(project="aslan", job_type="data-preprocessing",
                 notes="Extract all distances between texts in unseen")
run.use_artifact("CLINC150:v4")

embed = scipy.spatial.distance.pdist(dataset["embedding"], metric="cosine")

np.save("similar_data/unseen_distances.npy", embed)

my_data = wandb.Artifact("CLINC150-seen-similar", type="dataset")

my_data.add_dir("similar_data")
run.log_artifact(my_data)

run.finish()
