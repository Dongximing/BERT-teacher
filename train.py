import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from transformers import BertTokenizer, BertModel


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    # print(rounded_preds)
    # print(y)
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    # print(correct)
    # print(correct.sum())
    # print(len(correct))
    # print(acc)
    return acc


def train_fc(data_loader, model, optimizer, device, scheduler, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d["ids"]

        mask = d["mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)

        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float).unsqueeze(1)


        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask)



        loss = criterion(outputs, targets)
        acc = binary_accuracy(outputs, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()
    scheduler.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def eval_fc(valid_loader, model, device, criterion):
    model.eval()

    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            ids = d["ids"]

            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)

            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float).unsqueeze(1)

            outputs = model(ids=ids, mask=mask)
            loss = criterion(outputs, targets)

            # fin_targets.extend(targets.cpu().detach().numpy().tolist())
            # fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            acc = binary_accuracy(outputs, targets)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader)

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import time
# import random
# import numpy as np
# import os
# from tqdm import tqdm
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from torch.optim import lr_scheduler
# from transformers import BertTokenizer, BertModel
#
#
# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(torch.sigmoid(preds))
#
#     correct = (rounded_preds == y).float()  # convert into float for division
#     acc = correct.sum() / len(correct)
#
#     return acc
#
#
# def train_fc(iterator, optimizer,model,criterion,lr_scheduler):
#
#     model.train()
#     epoch_loss = 0
#     epoch_acc = 0
#     for  bi, batch in tqdm(enumerate(iterator), total=len(iterator)):
#
#         optimizer.zero_grad()
#
#         predictions = model(batch.reviews).squeeze(1)
#
#         loss = criterion(predictions, batch.label)
#
#         acc = binary_accuracy(predictions, batch.label)
#
#         loss.backward()
#
#         optimizer.step()
#
#         epoch_loss += loss.item()
#         epoch_acc += acc.item()
#     lr_scheduler.step()
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
#
#
#
#
# def eval_fc(iterator, model, criterion):
#     epoch_loss = 0
#     epoch_acc = 0
#
#     model.eval()
#
#     with torch.no_grad():
#         for bi, batch in tqdm(enumerate(iterator), total=len(iterator)):
#             predictions = model(batch.reviews).squeeze(1)
#
#             loss = criterion(predictions, batch.label)
#
#             acc = binary_accuracy(predictions, batch.label)
#
#             epoch_loss += loss.item()
#             epoch_acc += acc.item()
#
#     return epoch_loss / len(iterator), epoch_acc / len(iterator)
#
