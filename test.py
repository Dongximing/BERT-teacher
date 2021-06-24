import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from transformers import BertTokenizer, BertModel
from transformers import AdamW
import dataloader
import configs
import train
from model import BERTGRUSentiment
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

configs.seed_torch()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert = BertModel.from_pretrained('bert-base-uncased')
model = BERTGRUSentiment(bert,
                         configs.HIDDEN_DIM,
                         configs.OUTPUT_DIM,
                         configs.N_LAYERS,
                         configs.BIDIRECTIONAL,
                         configs.DROPOUT)
model.load_state_dict(torch.load('/home/dongx34/bert.pt'))
test_dataset = pd.read_csv('/home/dongx34/test.csv')
model.to(device)
test_dataset = dataloader.IMDBDataset(test_dataset['Reviews'].values, test_dataset['Sentiment'].values)
test_data_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False
)
criterion = nn.BCEWithLogitsLoss()
criterion = criterion.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def predict_sentiment(model, tokenizer, sentence):
    model.eval()
    sentence = " ".join(sentence.split())
    inputs = tokenizer.encode_plus(sentence, None,
                                   add_special_tokens=True,
                                   max_length=7,
                                   pad_to_max_length=True)

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    ids = torch.LongTensor(ids).to(device, dtype=torch.long).unsqueeze(0)
    mask = torch.LongTensor(mask).to(device, dtype=torch.long).unsqueeze(0)

    prediction = torch.sigmoid(model(ids=ids, mask=mask).to(device))
    return prediction.item()


valid_loss, valid_acc = train.eval_fc(test_data_loader, model, device, criterion)

print(valid_acc)
print("=----=")
print(valid_loss)
print("=----=")
print(predict_sentiment(model, tokenizer, "This film is terrible"))
print(predict_sentiment(model, tokenizer, "This film is good"))