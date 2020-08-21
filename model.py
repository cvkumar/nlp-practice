import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torch

import numpy as np

import tensorflow as tf

df = pd.read_csv("rt_reviews2.csv", engine="python")

df = pd.get_dummies(df, columns=["Freshness"], drop_first=True)

# !pip install tensorflow
df.iloc[2]["Review"]

df.iloc[0]

# TODO: Figure out how to add GPU
# Use cuda if present
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"

total_data_size = 50000
training_portion = 0.8
testing_portion = 0.2

training_slice = int(total_data_size * training_portion)

sentences = []
for sentence in df["Review"]:
    sentences.append(sentence)

labels = []
for label in df["Freshness_rotten"]:
    labels.append(label)

training_sentences = sentences[0:training_slice]
training_labels = labels[0:training_slice]

testing_sentences = sentences[training_slice:total_data_size]
testing_labels = labels[training_slice:total_data_size]

print(len(training_sentences))
print(len(testing_sentences))

# TODO: Use different tokenizer

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

# TODO: visualize data

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.summary()

# TODO: What are these epochs?
num_epochs = 15
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
)

sentence = ["Worst movie ever! Terrible from the start"]

sequence = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(
    sequence, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

result = model.predict(padded)

print(result)

if result[0] > 0.5:
    print("Negative Review")
else:
    print("Positive Review")
