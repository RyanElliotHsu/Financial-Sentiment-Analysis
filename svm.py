# -*- coding: utf-8 -*-
"""supervised_models.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K20155l9kBOzAjl8vZnAuDdbYMgn2zre

# Setting File Paths
"""

#mount google drive
#from google.colab import drive
#drive.mount('/content/drive')

#set file
# file_name = '/content/drive/MyDrive/Capstone/data/stocktwits_data.csv'
file2_name = '/scratch/reh424/data/clean_after_plug_1M.csv'
#dir_path = '/content/drive/MyDrive/Capstone/'

"""# Importing Libraries"""

import os
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from nltk.corpus import stopwords

"""# Data Inspecting"""

# df = pd.read_csv(file_name, usecols=['Message','Sentiment']).dropna(subset=['Sentiment'])
# #pd.set_option('display.max_rows', 15)
# df

df = pd.read_csv(file2_name,usecols=['body','sentiment'])
df.columns=['Message','Sentiment']

df['Sentiment'].value_counts()

"""# Data Preprocessing"""

#Setting all text to str type
df['Message'] = [i if type(i)==str else str(i) for i in df['Message']]

#set up dummy variables for classification
df = pd.get_dummies(df, columns=['Sentiment'])

# A list of contractions from http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

#snipped adapted from https://github.com/Idilismiguzel/NLP-with-Python/blob/master/Text-Classification.ipynb 
def clean_text(text, remove_stopwords = True):
    #Remove unwanted characters, stopwords, and format the text to create fewer nulls word embeddings
    # Convert words to lower case
    text = text.lower()
    
    # Replace contractions with their longer forms 
    if True:
        text = text.split()
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    # Tokenize each word
    text =  nltk.WordPunctTokenizer().tokenize(text)

    # # Lemmatize each token
    # lemm = nltk.stem.WordNetLemmatizer()
    # text = list(map(lambda word:list(map(lemm.lemmatize, word)), text))
    #commented out this part for now as it keeps slicing words into individual characters

    #remove first word - stock symbol
    if text:
      text.pop(0)
        
    return text

df['Message'] = [clean_text(i) for i in df['Message']]


"""# Train test split and bag of words"""

#train test split
train, test = train_test_split(df, train_size = 0.7, random_state=42)

#bag of words
bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=(3,3), lowercase=False)
X_tr_bow = bow_transform.fit_transform(train['Message'])
X_te_bow = bow_transform.transform(test['Message'])

y_tr = train['Sentiment_Bearish']
y_te = test['Sentiment_Bearish']

#Tf-ldf transformation
tfidf_transform = text.TfidfTransformer(norm=None)
X_tr_tfidf = tfidf_transform.fit_transform(X_tr_bow)

X_te_tfidf = tfidf_transform.transform(X_te_bow)

"""# Running Models"""

#Support vector machine
def svm_classify(X_tr, y_tr, X_test, y_test, description, _C=1.0):
    model = SVC(C=_C).fit(X_tr, y_tr)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    f1score = f1_score(y_test,y_pred)
    # pred_prob = model.predict_proba(X_test)
    # auc_score = roc_auc_score(y_test, pred_prob[:,1])
    print('Test Score with', description, 'features', score,'F1 Score:', f1score)
    return model

print("SVM MODEL:\n")
model_bow = svm_classify(X_tr_bow, y_tr, X_te_bow, y_te, 'bow')
model_tfidf = svm_classify(X_tr_tfidf, y_tr, X_te_tfidf, y_te, 'tf-idf')