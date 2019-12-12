import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import pickle as pkl

from mylib import preprocessing as prep

################################################################################
# Load data and preprocess
################################################################################
print("=== Preprocessing Data ===")
data_filename = "train.csv"
test_filename = "test.csv"
label = 'label'
sv_json = "trained_model.json"
sv_h5 = "model_weights.h5"

dtr = pd.read_csv(data_filename)
dpred = pd.read_csv(test_filename)

y = dtr['label'].to_numpy()
del dtr['label']
X = dtr.to_numpy()

# Normalize data set to 0-to-1 range
X = (pd.DataFrame(dtr).astype('float32'))
X_pred = (pd.DataFrame(dpred).astype('float32'))
X  /= 255
X_pred /= 255

print(X.shape, X_pred.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=0
)

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

################################################################################
# Define and set up model
################################################################################
print("=== Defining Model ===")
# Input Parameters
n_input = 784 # number of features
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dropout(0.3)(x)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

# Model have 6 layers : input layer, 4 hidden layer and 1 output layer
model = Model(Inp, output)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.summary()
################################################################################
# Training model
################################################################################
print("=== Start training ===")
# Hyperparameters
learning_rate = 0.1
adam = keras.optimizers.Adam(lr=learning_rate)

model.fit(
        X_train, y_train,
        batch_size = 100,
        epochs = 30,
        validation_data=(X_test, y_test),
        shuffle=True
    )
################################################################################
# Save structure and weights
################################################################################
print("=== Saving weights ===")
# Save neural network structure
model_structure = model.to_json()
f = Path("model_structure.json")
f.write_text(model_structure)

# Save neural network's trained weights
model.save_weights("model_weights.h5")
################################################################################
# Predict
################################################################################
print("=== Make prediction ===")
test_pred = pd.DataFrame(model.predict(X_pred, batch_size=200))
test_pred = pd.DataFrame(test_pred.idxmax(axis = 1))
test_pred.index.name = 'ImageId'
test_pred = test_pred.rename(columns = {0: 'Label'}).reset_index()
test_pred['ImageId'] = test_pred['ImageId'] + 1
test_pred.to_csv('submission.csv', index = False)
