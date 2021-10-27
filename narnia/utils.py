import torch
import numpy as np
import random
import os
import datetime
import logging

LOGGING_LEVEL = logging.INFO

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def append_prefix(path, prefix="artifacts"):
    return os.path.join(prefix, path)


def get_timestamp_str():
    return datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
