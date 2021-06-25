import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from transformers import BertTokenizer, BertModel
import dataloader
import configs
import train
from model import BERTGRUSentiment
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from functools import partial
configs.seed_torch()
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler



def train_titanic(config,checkpoint_dir=None,train_dir=None,valid_dir=None):
    # loading data
    train_dataset = pd.read_csv(train_dir)
    valid_dataset = pd.read_csv(valid_dir)
    train_dataset = dataloader.IMDBDataset(train_dataset['Reviews'].values,train_dataset['Sentiment'].values)
    valid_dataset = dataloader.IMDBDataset(valid_dataset['Reviews'].values,valid_dataset['Sentiment'].values)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle = True
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=int(config["batch_size"]), shuffle = False
    )


    # build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert = BertModel.from_pretrained('bert-base-uncased')
    model = BERTGRUSentiment(bert,
                         config["hidden_dim"],
                         configs.OUTPUT_DIM,
                         configs.N_LAYERS,
                         configs.BIDIRECTIONAL,
                         configs.DROPOUT)
    model.to(device)
#     for param in model.bert.parameters():
#       param.requires_grad = False


    #--------#
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    bert_identifiers = ['embedding', 'encoder', 'pooler']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': 3e-5 ,
         'betas': (0.9, 0.999),
         'weight_decay': 0.01 ,
         'eps': 1e-8},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': 3e-5,
         'betas': (0.9, 0.999),
         'weight_decay': 0.0,
         'eps': 1e-8},
        {'params': [param for name, param in model.named_parameters()
                    if not any(identifier in name for identifier in bert_identifiers)],
         'lr': CUSTOM_LEARNING_RATE,
         'betas': BETAS,
         'weight_decay': 0.0,
         'eps': 1e-8}
    ]
    optimizer = AdamW(grouped_model_parameters)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.01)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    best_loss = float('inf')
    for epoch in range (configs.EPOCHS):

        train_loss, train_acc = train.train_fc(train_data_loader, model, optimizer, device, lr_scheduler,criterion)

        valid_loss, valid_acc = train.eval_fc(valid_data_loader,model,device,criterion)


        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=valid_loss, accuracy=valid_acc)


#         if valid_loss < best_loss:
#             torch.save(model.state_dict(), configs.MODEL_PATH)
#             best_loss = valid_loss
#         print(f'Epoch: {epoch+1:02}')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

def main():
    max_num_epochs = 10
    num_samples =3

    train_dir = '/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/train.csv'
    valid_dir = '/home/dongxx/projects/def-mercer/dongxx/project/pythonProject/valid.csv'
    checkpoint_dir = configs.MODEL_PATH

    config = {
         "hidden_dim": tune.choice([128]),

         "batch_size": tune.choice([32,16])

    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=5,
        reduction_factor=2)
    reporter = CLIReporter(
        parameter_columns=["hidden_dim", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_titanic, checkpoint_dir=checkpoint_dir, train_dir=train_dir, valid_dir= valid_dir),
    
        resources_per_trial={"cpu": 4,"gpu":4},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")

    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))




if __name__ == "__main__":

    main()










# import torch
# import torch.nn as nn
# import torch.optim as optim
# import time
# import random
# import numpy as np
# import os
# import pandas as pd
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# from torch.optim import lr_scheduler
# from transformers import BertTokenizer, BertModel
# from transformers import AdamW
# import dataloader
# import config
# import train
# import spacy
# from model import BERTGRUSentiment
#
# from torchtext.legacy import data
# from torchtext.legacy import datasets
#
# from model import BERTGRUSentiment
# def seed_torch(seed=321):
#     random.seed(seed)
#     os.environ['PYTHONHASHSEED'] = str(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.enabled = False
#     np.random.RandomState(seed)
# seed_torch()
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
# def tokenize_and_cut(sentence):
#     tokens = tokenizer.tokenize(sentence)
#     tokens = tokens[:256 -2]
#     return tokens
# def run():
#
#     init_token_idx = tokenizer.cls_token_id
#     eos_token_idx = tokenizer.sep_token_id
#     pad_token_idx = tokenizer.pad_token_id
#     unk_token_idx = tokenizer.unk_token_id
#     max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']
#     TEXT = data.Field(batch_first = True,
#                   use_vocab = False,
#                   tokenize = tokenize_and_cut,
#                   preprocessing = tokenizer.convert_tokens_to_ids,
#                   init_token = init_token_idx,
#                   eos_token = eos_token_idx,
#                   pad_token = pad_token_idx,
#                   unk_token = unk_token_idx)
#
#     LABEL = data.LabelField(dtype=torch.float)
#     fields = {'Reviews':('reviews', TEXT),'Sentiment':('label', LABEL)}
#     train_data, valid_data = data.TabularDataset.splits(
#                             path = '/home/dongx34/',
#                             train = 'train.csv',
#                             validation = 'valid.csv',
#                             format = 'csv',
#                             fields = fields
# )
#     train_iterator, valid_iterator= data.BucketIterator.splits(
#     (train_data, valid_data),
#     sort = False, #don't sort test/validation data
#     batch_size=16,
#     device=device)
#     bert = BertModel.from_pretrained('bert-base-uncased')
#     model = BERTGRUSentiment(bert,
#                          config.HIDDEN_DIM,
#                          config.OUTPUT_DIM,
#                          config.N_LAYERS,
#                          config.BIDIRECTIONAL,
#                          config.DROPOUT)
#
#     LABEL.build_vocab(train_data)
#     optimizer = AdamW(model.parameters(),lr = 1e-5)
#     criterion = nn.BCEWithLogitsLoss()
#     model = model.to(device)
#     criterion = criterion.to(device)
#     best_loss = float('inf')
#     print(vars(train_data.examples[0]))
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.01)
#
#
#     for epoch in range (config.EPOCHS):
#
#         train_loss, train_acc = train.train_fc(train_iterator, optimizer, model,criterion,lr_scheduler )
#
#         valid_loss, valid_acc = train.eval_fc(valid_iterator,model,criterion)
#
#
#         if valid_loss < best_loss:
#             torch.save(model.state_dict(), config.MODEL_PATH)
#             best_loss = valid_loss
#         print(f'Epoch: {epoch+1:02}')
#         print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
#         print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
#
# #     print(LABEL.vocab.stoi)
# #
# #     print(vars(train_data[0]))
#
# #
# #     print(train_dataset.shape)
# #     train_dataset = dataloader.IMDBDataset(train_dataset['Reviews'].values,train_dataset['Sentiment'].values)
# #
# #     valid_dataset = dataloader.IMDBDataset(valid_dataset['Reviews'].values,valid_dataset['Sentiment'].values)
# #     test_dataset = dataloader.IMDBDataset(test_dataset['Reviews'].values,test_dataset['Sentiment'].values)
# #     train_data_loader = torch.utils.data.DataLoader(
# #         train_dataset, batch_size = 4, shuffle = True
# #     )
# #     valid_data_loader = torch.utils.data.DataLoader(
# #         valid_dataset, batch_size=4, shuffle = False
# #     )
# #
# #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #     bert = BertModel.from_pretrained('bert-base-uncased')
# #     model = BERTGRUSentiment(bert,
# #                          config.HIDDEN_DIM,
# #                          config.OUTPUT_DIM,
# #                          config.N_LAYERS,
# #                          config.BIDIRECTIONAL,
# #                          config.DROPOUT)
# #     model.to(device)
# #     optimizer = AdamW(model.parameters(), lr=3e-5)
# #     print(len(train_data_loader))
# #     for epoch in range (config.EPOCHS):
# #         train.train_fc(train_data_loader, model, optimizer, device)
# #
# #
#
# #     train_dataset = pd.read_csv('/home/dongx34/train.csv')
# #     valid_dataset = pd.read_csv('/home/dongx34/valid.csv')
# #     test_dataset = pd.read_csv('/home/dongx34/test.csv')
#
#
#
# if __name__ == "__main__":
#     run()
