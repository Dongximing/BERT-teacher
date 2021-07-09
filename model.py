import torch
import torch.nn as nn


class BERTGRUSentiment(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout):

        super().__init__()

        self.bert = bert

        embedding_dim = bert.config.to_dict()['hidden_size']
        self.LSTM = nn.LSTM(embedding_dim,hidden_dim,num_layers=n_layers,dropout=dropout, bidirectional=bidirectional,batch_first=True)

        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask):
        with torch.no_grad():

            embedded = self.dropout(self.bert(ids, attention_mask=mask)[0])

        # embedded = [batch size, sent len, emb dim]
        output,(hidden,ct) = self.LSTM(embedded)
        #_, hidden = self.rnn(embedded)
        # print(hidden.shape)

        # hidden = [n layers * n directions, batch size, emb dim]

        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]

        output = self.out(hidden)


        return output
# class BERTGRUSentiment(nn.Module):
#     def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):

#         super().__init__()

#         self.bert = bert
#         self.bert_drop = nn.Dropout(0.3)
#         self.out = nn.Linear(768, 1)

#     def forward(self, ids, mask, token_type_ids):
#         o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)[1]
#         bo = self.bert_drop(o2)
#         output = self.out(bo)
#         return output


# import torch
# import torch.nn as nn
#
# class BERTGRUSentiment(nn.Module):
#     def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):
#
#         super().__init__()
#
#         self.bert = bert
#
#         embedding_dim = bert.config.to_dict()['hidden_size']
#
#         self.rnn = nn.LSTM(embedding_dim,
#                           hidden_dim,
#                           num_layers=n_layers,
#                           bidirectional=bidirectional,
#                           batch_first=True,
#                           dropout=0 if n_layers < 2 else dropout)
#
#         self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
#
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, text,):
#
#         # text = [batch size, sent len]
#
#         with torch.no_grad():
#             embedded = self.bert(text)[0]
#
#         # embedded = [batch size, sent len, emb dim]
#
#         output, (hidden,h_c) = self.rnn(embedded)
#
#         # hidden = [n layers * n directions, batch size, emb dim]
#         # print(hidden.size())
#         if self.rnn.bidirectional:
#             hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
#         else:
#             hidden = self.dropout(hidden[-1, :, :])
#
#         # hidden = [batch size, hid dim]
#
#         output = self.out(hidden)
#
#         # output = [batch size, out dim]
#
#         return output