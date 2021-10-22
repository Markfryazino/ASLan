from transformers import BertTokenizerFast, BertModel, DataCollatorWithPadding, AdamW, get_scheduler
from tqdm.auto import tqdm, trange
import torch
import wandb
from datasets import load_dataset, ClassLabel, load_metric, DatasetDict
import numpy as np


np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained("bert-base-uncased").to("cuda")

dataset = DatasetDict.load_from_disk("datasets/data")
collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

dataset["DtrainStrain"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtrainStrain"] = dataset["DtrainStrain"].shuffle()

dataset["DtrainSval"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtrainSval"] = dataset["DtrainSval"].shuffle()

dataset["DtrainStest"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtrainStest"] = dataset["DtrainStest"].shuffle()

dataset["DtestStrain"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtestStrain"] = dataset["DtrainStrain"].shuffle()

dataset["DtestSval"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtestSval"] = dataset["DtrainSval"].shuffle()

dataset["DtestStest"].set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
dataset["DtestStest"] = dataset["DtrainStest"].shuffle()


model = BertProtoNet(bert).to("cuda")

config = {
    "class_number": 30,
    "support_size": 5,
    "query_size": 10,
    "labels": dataset["DtrainStrain"]["label"].unique()
}

num_episodes = 5000
log_freq = 10
eval_freq = 100
eval_num = 100

run = wandb.init(project="aslan",
                 tags=["protonet", "clinc150"],
                 job_type="training",
                 config=config)

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_episodes
)
loss = torch.nn.CrossEntropyLoss()
wandb.watch(model, loss)

train_acc = 0
train_loss = 0
for episode in trange(1, num_episodes + 1):
    model.train()
    support, query = sample_episode(dataset["DtrainStrain"], **config)
    support_pt, query_pt, support_labels, query_labels = prepare_inputs(support, query, collator)
    outputs = model(support_pt, support_labels, query_pt, batch_size=32)

    mapping = {val.item(): key for key, val in enumerate(outputs["labels"])}
    rebranded_query_labels = torch.LongTensor([mapping[el.item()] for el in query_labels]).cuda()

    loss_val = loss(-outputs["distances"], rebranded_query_labels)
    loss_val.backward()
    
    train_acc += (rebranded_query_labels == outputs["probas"].argmax(axis=1)).sum().item() / len(rebranded_query_labels)
    train_loss += loss_val.item()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    if episode % log_freq == 0:
        wandb.log({"episode": episode, "train/accuracy": train_acc / log_freq, "train/loss": train_loss / log_freq})
        train_acc = 0
        train_loss = 0
    
    if episode % eval_freq == 0:
        model.eval()
        with torch.no_grad():
            eval_acc = 0.
            eval_loss = 0.
            for i in range(eval_num):
                support, query = sample_episode(dataset["DtestStrain"], **config, query_dataset=dataset["DtestStest"])
                support_pt, query_pt, support_labels, query_labels = prepare_inputs(support, query, collator)
                outputs = model(support_pt, support_labels, query_pt, batch_size=32)

                mapping = {val.item(): key for key, val in enumerate(outputs["labels"])}
                rebranded_query_labels = torch.LongTensor([mapping[el.item()] for el in query_labels]).cuda()

                loss_val = loss(-outputs["distances"], rebranded_query_labels)
                eval_acc += (rebranded_query_labels == outputs["probas"].argmax(axis=1)).sum().item() / len(rebranded_query_labels)      
                eval_loss += loss_val.item()     
            wandb.log({"episode": episode, "test/accuracy": eval_acc / eval_num, "test/loss": eval_loss / eval_num})
            

model.model.save_pretrained("protonet")

my_data = wandb.Artifact("Protonet1", type="model", description="First pretrained Prototypical Network")

my_data.add_dir("protonet")
run.log_artifact(my_data)

wandb.finish()
