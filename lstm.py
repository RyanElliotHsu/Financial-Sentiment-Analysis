#set file
file_name = '/scratch/reh424/data/preprocessed.csv'
embedding = '/scratch/reh424/data/glove.6B.50d.txt'

"""# Importing Libraries"""

import os
import pandas as pd
import numpy as np
import re

"""# Data Inspecting"""

df = pd.read_csv(file_name)
df = df.drop('Unnamed: 0', axis=1)

"""# Word Embeddings"""

words = dict()

def add_to_dict(d, filename):
  with open(filename, 'r') as f:
    for line in f.readlines():
      line = line.split(' ')

      try:
        d[line[0]] = np.array(line[1:], dtype=float)
      except:
        continue

add_to_dict(words, embedding)

print("one")

def message_to_word_vectors(message, word_dict=words):
  processed_list_of_tokens = [t for t in message if t in words]

  vectors = []

  for token in processed_list_of_tokens:
    if token not in word_dict:
      continue
    
    token_vector = word_dict[token]
    vectors.append(token_vector)
  
  return np.array(vectors, dtype=float)

"""# Train Test Split"""

df = df.sample(frac=1, random_state=1)
df.reset_index(drop=True, inplace=True)

split_index_1 = int(len(df) * 0.7)
split_index_2 = int(len(df) * 0.85)

train_df, val_df, test_df = df[:split_index_1], df[split_index_1:split_index_2], df[split_index_2:]

len(train_df), len(val_df), len(test_df)

print("two")

def df_to_X_y(dff):
  y = dff['Sentiment_Bullish'].to_numpy().astype(int)

  all_word_vector_sequences = []

  for message in dff['Text']:
    message_as_vector_seq = message_to_word_vectors(message)
    
    if message_as_vector_seq.shape[0] == 0:
      message_as_vector_seq = np.zeros(shape=(1, 50))

    all_word_vector_sequences.append(message_as_vector_seq)
  
  return all_word_vector_sequences, y

X_train, y_train = df_to_X_y(train_df)

print("three")

print(len(X_train), len(X_train[0]))

print(len(X_train), len(X_train[4]))

sequence_lengths = []

for i in range(len(X_train)):
  sequence_lengths.append(len(X_train[i]))

pd.Series(sequence_lengths).describe()

print("four")

from copy import deepcopy

def pad_X(X, desired_sequence_length=956):
  X_copy = deepcopy(X)

  for i, x in enumerate(X):
    x_seq_len = x.shape[0]
    sequence_length_difference = desired_sequence_length - x_seq_len
    
    pad = np.zeros(shape=(sequence_length_difference, 50))

    X_copy[i] = np.concatenate([x, pad])
  
  return np.array(X_copy).astype(float)

X_train = pad_X(X_train)

print("five")

X_train.shape

y_train.shape

X_val, y_val = df_to_X_y(val_df)
X_val = pad_X(X_val)

X_val.shape, y_val.shape

X_test, y_test = df_to_X_y(test_df)
X_test = pad_X(X_test)

X_test.shape, y_test.shape

"""# LSTM Model"""

# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# model = Sequential([])

# model.add(layers.Input(shape=(956, 50)))
# model.add(layers.LSTM(64, return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(64, return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.LSTM(64, return_sequences=True))
# model.add(layers.Dropout(0.2))
# model.add(layers.Flatten())
# model.add(layers.Dense(1, activation='sigmoid'))

# model.summary()

# from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import AUC
# from tensorflow.keras.callbacks import ModelCheckpoint

# cp = ModelCheckpoint('model/', save_best_only=True)

# model.compile(optimizer=Adam(learning_rate=0.0001), 
#               loss=BinaryCrossentropy(), 
#               metrics=['accuracy', AUC(name='auc')])

# frequencies = pd.value_counts(train_df['Sentiment_Bullish'])

# frequencies

# weights = {0: frequencies.sum() / frequencies[0], 1: frequencies.sum() / frequencies[1]}
# weights

# model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, callbacks=[cp], class_weight=weights)

# from tensorflow.keras.models import load_model

# best_model = load_model('model/')

# test_predictions = (best_model.predict(X_test) > 0.5).astype(int)

# from sklearn.metrics import classification_report

# print(classification_report(y_test, test_predictions))