import torch
import numpy as np

from datasets import concatenate_datasets, load_dataset
from transformers import DataCollatorWithPadding


class BertProtoNet(torch.nn.Module):
    def __init__(self, bert_model):
        super(BertProtoNet, self).__init__()
        self.model = bert_model

    def forward(self, support_set, support_labels, query_set, batch_size=-1):
        device = self.model.device

        if batch_size == -1:
            support_embeddings = self.model(**support_set)[1]
            query_embeddings = self.model(**query_set)[1]
        else:
            support_embeddings = torch.cat([self.model(**{key: val[i:i + batch_size] for key, val in 
                                                         support_set.items()})[1] for i in range(0, support_set["input_ids"].size(0), batch_size)])
            query_embeddings = torch.cat([self.model(**{key: val[i:i + batch_size] for key, val in
                                                        query_set.items()})[1] for i in range(0, query_set["input_ids"].size(0), batch_size)])

        labels = torch.unique(support_labels)
        centroids = torch.FloatTensor(size=(labels.size()[0], support_embeddings.size()[1])).to(device)

        for i, label in enumerate(labels):
            centroids[i] = support_embeddings[support_labels == label].mean()

        batched_centroids = torch.unsqueeze(centroids, 0)
        batched_queries = torch.unsqueeze(query_embeddings, 0)

        distances = torch.cdist(batched_queries, batched_centroids, p=2)[0]
        return {
            "probas": torch.nn.functional.softmax(-distances, dim=1),
            "distances": distances,
            "labels": labels
        }


def sample_idxs(dataset, label, num):
    idxs = torch.nonzero(dataset["label"] == label).reshape(-1)
    batch_idxs = np.random.choice(idxs, num)
    np.random.shuffle(batch_idxs)
    return batch_idxs


def sample_episode(dataset, class_number, support_size, query_size, labels, query_dataset=None):
    episode_classes = np.random.choice(labels, class_number)

    supports, queries = [], []
    for label in episode_classes:
        if query_dataset is None:
            batch_idxs = sample_idxs(dataset, label, support_size + query_size)
            supports.append(dataset.select(batch_idxs[:support_size]))
            queries.append(dataset.select(batch_idxs[support_size:]))
        else:
            supports.append(dataset.select(sample_idxs(dataset, label, support_size)))
            queries.append(query_dataset.select(sample_idxs(query_dataset, label, query_size)))
    
    return concatenate_datasets(supports), concatenate_datasets(queries)


def prepare_inputs(support, query, collator, device="cuda"):
    support_pt = collator(support[:]).to(device)
    query_pt = collator(query[:]).to(device)

    support_labels = support_pt["labels"].to(device)
    del support_pt["labels"]

    if "labels" in query_pt:
        query_labels = query_pt["labels"].to(device)
        del query_pt["labels"]
    else:
        query_labels = None
    
    return support_pt, query_pt, support_labels, query_labels
