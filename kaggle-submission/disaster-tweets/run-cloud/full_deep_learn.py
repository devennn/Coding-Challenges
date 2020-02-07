import pandas as pd
from pathlib import Path
import os
import re
from unicodedata import normalize
import string
import pickle as pkl
import os
# Disable all warning include tensorflow gpu debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
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
        # Remove whitespace at start of sentence
        tweet = tweet.strip()
        temp.append(tweet)
    try:
        # Concatenate training with target
        processed_tweets = pd.concat([pd.DataFrame(temp), df['target']], axis=1)
        processed_tweets = pd.DataFrame(processed_tweets)
    except KeyError:
        processed_tweets = pd.DataFrame(temp)
    print(processed_tweets)
    return processed_tweets

def vectorize_tweets(count_vect, data):
    vect_tweets = count_vect.fit_transform(data)
    vect_tweets = vect_tweets.toarray()
    return vect_tweets, count_vect

def define_model(input_len, output_len):
    n_hidden_1 = math.ceil(input_len / 2)
    n_hidden_2 = math.ceil(n_hidden_1 / 2)
    n_hidden_3 = n_hidden_2
    n_hidden_4 = math.ceil(input_len / 2)

    Inp = Input(shape=(input_len, ))
    x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
    x = Dropout(0.3)(x)
    x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
    x = Dropout(0.3)(x)
    x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
    x = Dropout(0.3)(x)
    x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
    output = Dense(output_len, activation='softmax', name = "Output_Layer")(x)

    model = Model(Inp, output)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.summary()
    return model

def train(model, X_train, X_test, y_train, y_test):
    # Hyperparameters
    learning_rate = 0.1
    adam = keras.optimizers.Adam(lr=learning_rate)

    model.fit(
            X_train, y_train,
            batch_size = 32,
            epochs = 1,
            validation_data=(X_test, y_test),
            shuffle=True
        )
    return model

def save_submission(new_prediction, fname):
    new_prediction = new_prediction.rename({0: 'target'}, axis=1)
    new_prediction.to_csv(fname, index=False)

if __name__ == '__main__':
    path = Path('.').parent.absolute()

    full_train = os.path.join(path, 'raw-dataset', 'train.csv')
    train_df = pd.read_csv(full_train, encoding='utf-8')

    full_test = os.path.join(path, 'raw-dataset', 'test.csv')
    test_df = pd.read_csv(full_test, encoding='utf-8')

    # Preprocess training and testing tweets
    processed_tr_tweets = cleaning(train_df['text'], train_df)
    processed_tst_tweets = cleaning(test_df['text'], test_df)

    # Convert a collection of text documents to a matrix of token counts
    count_vect = CountVectorizer(analyzer='word', lowercase=False, stop_words='english')
    # Combine both train and test
    # Prevent unequal length of variables after tokenization
    combined_tr_tst = pd.concat([processed_tr_tweets[0], processed_tst_tweets[0]], axis=0)
    combined_vect,_ = vectorize_tweets(count_vect, combined_tr_tst)

    # Check length
    len_tr = len(processed_tr_tweets[0])
    print('Training length: %d' %len_tr)
    len_tst = len(processed_tst_tweets[0])
    print('Testing length: %d' %len_tst)
    print('Length of train + test: %d' %len(combined_vect))

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

    input_len = X_train.shape[1]
    model = define_model(input_len, 2)

    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()
    model = train(model, X_train, X_test, y_train_np, y_test_np)

    # Predict
    dl_predictions = pd.DataFrame(model.predict(vect_tst_tweets))
    dl_rounded = pd.DataFrame([int(x) for x in dl_predictions[1]])
    print(dl_rounded)

    format_predictions = pd.concat([test_df['id'], dl_rounded], axis=1)
    print(format_predictions)

    save_submission(format_predictions, 'submission3.csv')
