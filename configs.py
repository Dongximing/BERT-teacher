import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from transformers import BertTokenizer, BertModel
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
EPOCHS = 8
MODEL_PATH ='/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/checkpoints/'
