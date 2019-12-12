import os
# Disable all warning include tensorflow gpu debug
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.compat.v1.keras.models import model_from_json
from pathlib import Path
import numpy as np
import pandas as pd

################################################################################
# Load model and weights
################################################################################
print("=== Load structure and weights ===")
json_model = "model_structure.json"
h5_weights = "model_weights.h5"

# Load the json file that contains the model's structure
f = Path(json_model)
model_structure = f.read_text()
# Recreate the Keras model object from the json data
model = model_from_json(model_structure)
# Re-load the model's trained weights
model.load_weights(h5_weights)

################################################################################
# Load data to predict
test_filename = "test.csv"
dpred = pd.read_csv(test_filename)
del dpred['id']
X_pred = (pd.DataFrame(dpred).astype('float32'))
X_pred /= 255
################################################################################

################################################################################
# Predict
################################################################################
print("=== Make prediction ===")
id_col = 'id'
label_col = 'label'

predictions = pd.DataFrame(model.predict(X_pred, batch_size=200))
predictions = pd.DataFrame(predictions.idxmax(axis = 1))
predictions.index.name = id_col
predictions = predictions.rename(columns = {0: label_col}).reset_index()
predictions[id_col] = predictions[id_col]
predictions.to_csv('submission.csv', index = False)
