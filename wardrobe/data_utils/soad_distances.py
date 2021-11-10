import wandb
import numpy as np
import torch
import json
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict, concatenate_datasets
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import scipy
from tqdm.notebook import trange, tqdm
import scipy.spatial.distance


DATASETS = ["BANKING77", "CLINC150", "HWU64", "SNIPS", "ATIS", "NLUED"]
SPLITS = ["train", "valid", "test"]

def load_from_memory(dataset, root_path="artifacts/SOAD:v2"):
    mapping = {el: os.path.join(root_path, dataset, el + ".csv") for el in ["train", "valid", "test"]}
    mapping["val"] = mapping["valid"]
    del mapping["valid"]

    raw = load_dataset("csv", data_files=mapping)
    return concatenate_datasets([raw["train"], raw["val"], raw["test"]])

embedder = SentenceTransformer('all-mpnet-base-v2').cuda()
root_path="artifacts/SOAD:v2"

for dataset in tqdm(DATASETS):
    data = load_from_memory(dataset)
    data = data.map(lambda x: {"embedding": embedder.encode(x["text"]), **x}, batched=True, load_from_cache_file=False)
    embed = scipy.spatial.distance.pdist(data["embedding"], metric="cosine")
    np.save(os.path.join(root_path, dataset, "distances.npy"), embed)
