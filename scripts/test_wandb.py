print("hey")

import wandb
import os

os.environ["WANDB_MODE"] = "offline"

wandb.init(
    entity="broccoliman",
    project="aslan",
    tags=["debug"]
)

print("init happened")
wandb.log({"hey": "heyhey"})
print("log happened")
wandb.finish()
print("finished")