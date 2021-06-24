import pandas as pd
import torch
import numpy as np
#from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn import model_selection
import  nltk
nltk.download('stopwords')
train_data = pd.read_excel('/home/dongxx/projects/def-mercer/dongxx/project/data/IMDB-Movie-Reviews-Large-Dataset-50k/train.xlsx')
testing_data = pd.read_excel('/home/dongxx/projects/def-mercer/dongxx/project/data/IMDB-Movie-Reviews-Large-Dataset-50k/test.xlsx')
import configs
SEED =42
stop = stopwords.words('english')
configs.seed_torch()
def data_cleaning(train_data):
    train_data['Reviews'] = train_data['Reviews'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
    train_data['Reviews'] = train_data['Reviews'].str.replace('[^\w\s]','')
    # train_data['Reviews'] = train_data['Reviews'].apply(lambda x: " ".join(x for x in str(x).split() if x not in stop))
    # train_data['Reviews'] = train_data['Reviews'].apply(lambda x: str(TextBlob(x).correct()))
    train_data['Sentiment'] = train_data['Sentiment'].apply(lambda x:1 if x=="pos" else 0)
data_cleaning(train_data)
data_cleaning(testing_data)

train_data, valid_data = model_selection.train_test_split(train_data,test_size=0.2,random_state=SEED,stratify=train_data['Sentiment'].values)
train_data= train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
print(train_data.shape)
train_data.to_csv('train.csv', encoding='utf-8')
valid_data.to_csv('valid.csv', encoding='utf-8')

testing_data.to_csv('test.csv', encoding='utf-8')

print(train_data.info())
print(valid_data.info())
print(testing_data.info())
