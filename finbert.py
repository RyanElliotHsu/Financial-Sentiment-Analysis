#import libraries
import os
import pandas as pd
import numpy as np
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.metrics import f1_score

file_name = 'stocktwits_data.csv'

df = pd.read_csv(file_name, usecols=['Message','Sentiment']).dropna(subset=['Sentiment'])
df['Message'] = [i if type(i)==str else str(i) for i in df['Message']]
text = df['Message'].to_list()

finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

def get_sentiment(input):
  inputs = tokenizer(input, return_tensors="pt", padding=True)
  outputs = finbert(**inputs)
  predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
  return predictions

pred = get_sentiment(text)

#remove neutral column from tensor
pred = pred[:,[1,2]]

#get max from positive and negative columns as tensor
pred_binary = torch.argmax(pred,dim=1)

df['predictions'] = pred_binary.tolist()

df.to_csv('finbert.csv')