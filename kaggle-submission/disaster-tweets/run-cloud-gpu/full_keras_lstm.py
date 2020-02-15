import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
from unicodedata import normalize
import string
import pickle as pkl
import os
import sys
# Disable all warning include tensorflow gpu debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk
words = set(nltk.corpus.words.words())

from keras import Sequential
from keras.layers import *
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import math

def cleaning(tweet_text, df):
    temp = []
    table = str.maketrans("", "", string.punctuation)
    for tweet in tweet_text:
        # Remove links
        tweet = re.sub(r"http\S+", "", tweet)
        # Remove newline
        tweet = tweet.strip('\n')
        # Remove unicode
        tweet = normalize('NFKD', tweet).encode('ascii','ignore')
        # Remove username
        tweet = re.sub('@[^\s]+','',str(tweet))
        # Remove punctuation and change to lower case
        tweet = tweet.translate(table).lower()
        # Remove 'b' at the begining for binary
        tweet = tweet.replace('b', '', 1)
        # # Remove number
        # tweet = ''.join(i for i in tweet if not i.isdigit())
        # Remove non english
        tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) if w.lower()
            in words or not w.isalpha())
        temp.append(tweet)
    try:
        # Concatenate training with target
        processed_tweets = pd.concat([pd.DataFrame(temp), df['target']], axis=1)
        processed_tweets = pd.DataFrame(processed_tweets)
    except KeyError:
        processed_tweets = pd.DataFrame(temp)
    print(processed_tweets)
    return processed_tweets

def vectorize_tweets(tokenizer, data):
    tokenizer.fit_on_texts(data)
    vect_tweets = tokenizer.texts_to_sequences(data)
    vect_tweets = pad_sequences(vect_tweets)
    return vect_tweets, tokenizer

def define_model(input_len, output_len, input_dim):
    embed_dim = 128
    lstm_out = 200

    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=embed_dim,
        input_length=input_len)
    )
    # model.add(Dropout(0.2))
    model.add(LSTM(lstm_out, dropout=0.2))
    model.add(Dense(output_len, activation='softmax'))

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train(model, X_train, X_test, y_train, y_test):

    model.fit(
            X_train, y_train,
            batch_size = 32,
            epochs = 10,
            validation_data=(X_test, y_test),
            shuffle=True
        )
    save_structure_weight(model)
    return model

def save_structure_weight(model):
    print("Saving Model...")
    model_structure = model.to_json()
    with open('model_structure.json', 'w') as f:
        f.write(model_structure)

    print("Saving Weights...")
    model.save_weights('model_weights.h5')

def save_submission(new_prediction, fname):
    new_prediction = new_prediction.rename({0: 'target'}, axis=1)
    new_prediction.to_csv(fname, index=False)

if __name__ == '__main__':
    path = Path('.').parent.absolute()
    print('=== Reading data from {} ==='.format(path))
    full_train = os.path.join(path, 'raw-dataset', 'train.csv')
    train_df = pd.read_csv(full_train, encoding='utf-8')

    full_test = os.path.join(path, 'raw-dataset', 'test.csv')
    test_df = pd.read_csv(full_test, encoding='utf-8')

    print('=== Cleaning texts ===')
    # Preprocess training and testing tweets
    processed_tr_tweets = cleaning(train_df['text'], train_df)
    processed_tst_tweets = cleaning(test_df['text'], test_df)

    print('=== Tokenizing texts ===')
    # Convert a collection of text documents to a matrix of token counts
    tokenizer = Tokenizer()
    # Combine both train and test
    # Prevent unequal length of variables after tokenization
    combined_tr_tst = pd.concat([processed_tr_tweets[0], processed_tst_tweets[0]], axis=0)
    combined_vect, _ = vectorize_tweets(tokenizer, combined_tr_tst)

    # Check length
    len_tr = len(processed_tr_tweets[0])
    print('Training length: %d' %len_tr)
    len_tst = len(processed_tst_tweets[0])
    print('Testing length: %d' %len_tst)
    print('Length of train + test: %d' %len(combined_vect))

    print('=== Split to train and test data ===')
    # Split back to train and test
    vect_tweets = combined_vect[:len_tr]
    vect_tst_tweets = combined_vect[len_tr:]

    # Split training and testing
    X_train, X_test, y_train, y_test  = train_test_split(
            vect_tweets,
            processed_tr_tweets['target'],
            train_size=0.80,
            random_state=True,
            shuffle=True
    )

    print('=== Defining Model ===')
    input_dim = len(tokenizer.word_index) + 1
    input_len = len(combined_vect[0])
    output_len = len(set(y_train))
    print('Input Length: {}\nOutput Length: {}\nInput_dim: {}'.format(
            input_len, output_len, input_dim
        ))
    model = define_model(input_len, output_len, input_dim)

    print('=== Training Model ===')
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)
    model = train(model, X_train, X_test, y_train_np, y_test_np)

    print('=== Perform predictions ===')
    dl_predictions = pd.DataFrame(model.predict(vect_tst_tweets))
    dl_rounded = pd.DataFrame([round(x) for x in dl_predictions[1]])
    print(dl_rounded)

    format_predictions = pd.concat([test_df['id'], dl_rounded], axis=1)
    print(format_predictions)

    print('=== Save predictions to submission3.csv ===')
    save_submission(format_predictions, 'submission3.csv')
