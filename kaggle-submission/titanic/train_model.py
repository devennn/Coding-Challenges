import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import linear_model
from sklearn import ensemble
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import utils
import joblib as joblib
import os

"""
Load data and setup features and label

Arguments: data_filename which is the file name of formatted data
Return: X as featurs and y as labels of data
"""
def load_data(data_filename, label, trained_model_filename):
    print("=== Loading Data ===")
    # Load the data set
    df = pd.read_csv(data_filename)

    # Create the X and y arrays
    y = df[label].to_numpy()
    del df[label]
    X = df.to_numpy()

    # Remove trainde model fle if exists
    if os.path.exists(trained_model_filename):
        print("Deleting Old file..." + trained_model_filename)
        os.remove(trained_model_filename)

    return X, y

"""
Setting up model parameters based.

Arguments: X as features and y as labels
Return: tupels of model, X as labels for training and testing and
        y as label for training and testing
"""
def setup_data_and_model(X, y):
    print("=== Setting up data ===")
    # Split the data set in a training set (70%) and a test set (30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0
    )

    print("=== Setting up model ===")
    model = linear_model.LogisticRegression(solver='lbfgs')
    return model, X_train, X_test, y_train, y_test

"""
Train model from test data

Arguments: Take model, X_train and y_train for training data and X_test and
           y_test for testing data
Return: model that has been trained and predictions
"""
def train_model(model, X_train, X_test, y_train, y_test):
    # Train and save model
    for i in range(3):
        utils.shuffle(X_train, y_train)
        print("=== Training In progress ===")
        print("Iteration : " + str(i))
        model.fit(X_train, y_train)
        utils.shuffle(X_test, y_test)
        predictions = model.predict(X_test)
        # Print Error rate
        print("=== Accuracy Score ===")
        accuracy = accuracy_score(y_test, predictions)
        print(accuracy)
    print("=== Done Training ===")
    return model, predictions

"""
Visualize performance of model

Arguments: Take predictions and y test label
Return: Nothing
"""
def visualize_performance(predictions, y_test):
    print("=== visualize performance ===")
    conf_matrix = confusion_matrix(y_test, predictions)
    performance = pd.DataFrame(
                    conf_matrix,
                    columns=['Survived', 'Died'],
                    index=[['Survived', 'Died']]
                )
    print(performance)

"""
Perform cross validation to double check model accuracy

Arguments: Take all X features and all y labels data
Return: Nothing
"""
def perform_cross_validation(X, y, model):
    print("=== Perform cross validation ===")
    scores = cross_val_score(model, X, y, cv = 10)
    mean_scores = np.mean(scores)
    print(mean_scores)
    return model

"""
Main function
"""
def main():
    data_filename = 'train_formatted.csv'
    label = 'Survived'
    trained_model_filename = 'titanic_model.pkl'

    # Load data
    X, y = load_data(data_filename, label, trained_model_filename)
    # Setup data and model
    model, X_train, X_test, y_train, y_test = setup_data_and_model(X, y)
    # Train
    model, predictions = train_model(model, X_train, X_test, y_train, y_test)
    # Check performance
    visualize_performance(predictions, y_test)
    model = perform_cross_validation(X, y, model)
    # Save model
    print("=== Training done. Saving model ===")
    joblib.dump(model, trained_model_filename)
    print("=== Done Everything ===")

if __name__ == '__main__':
    main()
