import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os

def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    np.random.RandomState(seed)




OUTPUT_DIM =1
N_LAYERS =2
BIDIRECTIONAL = True
DROPOUT = 0.25
EPOCHS = 15
MODEL_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/bert.pt'
