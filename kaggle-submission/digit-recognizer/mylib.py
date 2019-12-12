"""

This module contains helper functions for preprocessing, training and predictions

"""
# Required module
import pandas as pd
import numpy as np
import webbrowser
import os
import matplotlib.pyplot as plt

from warnings import simplefilter
# ignore all future warnings
def activate_warning_surpressor():
    simplefilter(action='ignore', category=FutureWarning)
    simplefilter(action='ignore', category=UserWarning)
    simplefilter(action='ignore', category=RuntimeWarning)

# Scikit leearn algorithm
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors
from sklearn import discriminant_analysis as da
from sklearn import naive_bayes
from sklearn import linear_model

# Scikit learn others
from sklearn import utils
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

################################################################################
# Start preprocessing classs
################################################################################
class preprocessing():
    """
    Load data and setup features and label

    Arguments: data_filename which is the file name of formatted data
    Return: X as featurs and y as labels of data
    """
    def load_training_data(data_filename, label, trained_model_filename):
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
    View new data on webpage

    Arguments:
        - data is the Data to view
    Return:
        - Nothing
    """
    def view_on_html(data):
        html = data[0:100].to_html()
        with open("deleteme.html", "w") as f:
            f.write(html)
        full_filename = os.path.abspath("deleteme.html")
        err = webbrowser.open("file://{}".format(full_filename))

    """
    Perform quick check of the number of NaN data

    Arguments:
        - df_list is the list of dataframe to be viewed
    Return:
        - Nothing
    """
    def quick_overview(df_list):
        for i in range(len(df_list)):
            df = df_list[i]
            rows = len(df.index)
            print(
                '\n============================\n' +
                '::: List ' + str(i) + ' :::' +
                '\n============================\n' +
                'dtypes' +
                '\n============================\n' +
                str(df.dtypes) +
                '\n============================\n' +
                'Number of Null in Every Column' +
                '\n============================\n' +
                str(df.apply(lambda x: rows - x.count(), axis=0))
            )

    """
    Change old column data to new. Numeric only. If delOldCol is True,
    the oldCol wil be deleted

    Arguments:
        - df as dataframe
        - cutRange is the list of range eg: [0, 1] low = 0, high = 1
        - dataLabel is the label to put into new column
        - oldCol is the name of column to replace
        - newCol is the name of new column
        - delOldCol is a bool to that indicate whether to delete oldCol or not
    Return:
        - df is dataframe
    """
    def replace_column_numeric(df, cutRange, dataLabel, oldCol, newCol, delOldCol):
        df[newCol] = pd.cut(
                        df[oldCol],
                        cutRange,
                        labels=dataLabel
                    )
        if(delOldCol == True):
            del df[oldCol]
        return df

    """
    Delete all unwanted Columns

    Arguments:
        - df is the dataframe
        - col_name is the list of column to delete
    Return:
        - Nothing
    """
    def delete_unwanted_columns(df, col_name):
        for col in col_name:
            del df[col]
        return df

################################################################################
# End of preprocessing Class
################################################################################

################################################################################
# Start Training Class
################################################################################
class training():
    """
    Rough test of algorithm. Used to see which one gives the best value

    Arguments:
        - X_train, X_test, y_train, y_test as the training and testing data
    Return:
        - Nothing
    """
    def test_classifier(X_train, X_test, y_train, y_test):
        # Surpress warnings
        activate_warning_surpressor()
        # Algorithms that will be compared
        algorithm = [
            ensemble.RandomForestClassifier(),
            ensemble.AdaBoostClassifier(),
            ensemble.GradientBoostingClassifier(),
            svm.SVC(),
            svm.LinearSVC(),
            svm.NuSVC(),
            neighbors.KNeighborsClassifier(3),
            da.LinearDiscriminantAnalysis(),
            da.QuadraticDiscriminantAnalysis(),
            naive_bayes.GaussianNB(),
            linear_model.LogisticRegression(solver='lbfgs')
        ]
        # Run model and pring metrics
        for i in range(len(algorithm)):
            print("=" * 40)
            print("Running : " + algorithm[i].__class__.__name__)
            model = algorithm[i]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print("Accuracy: {:.4%}".format(accuracy))
            mse = mean_absolute_error(y_train, model.predict(X_train))
            print("Mean Abs Error train : %.4f" % mse)
            mse = mean_absolute_error(y_test, model.predict(X_test))
            print("Mean Abs Error test  : %.4f" % mse)

    """
    Train model from test data

    Arguments:
        - model
        - data that contains X_train, y_train for training data and X_test and
        y_test for testing data
        - epochs is the number of iteration
    Return:
        - model that has been trained
        - predictions
    """
    def train_model(model, data, epochs=1):
        X_train = data[0]
        X_test = data[1]
        y_train = data[2]
        y_test = data[3]
        # Train and save model
        for i in range(epochs):
            utils.shuffle(X_train, y_train)
            print("=== Training In progress ===")
            print("Epoch(s) : " + str(i))
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
    Perform cross validation to double check model accuracy

    Arguments:
        - X features
        - y labels data
        - model training
        - cv int
    Return:
        - Nothing
    """
    def perform_cross_validation(X, y, model, cv=10):
        print("=== Perform cross validation ===")
        scores = cross_val_score(model, X, y, cv=cv)
        mean_scores = np.mean(scores)
        print(mean_scores)
        return model

    """
    Visualize performance of model

    Arguments:
        - predictions
        - y_test data
        - label list for confusion matrix
    Return:
        - Nothing
    """
    def view_confusion_matrix(predictions, y_test, label):
        print("=== Confusion Matrix ===")
        conf_matrix = confusion_matrix(y_test, predictions)
        performance = pd.DataFrame(conf_matrix, columns=label, index=[label])
        print(performance)
