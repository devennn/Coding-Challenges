import pandas
import joblib as joblib
import pickle as pkl
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, accuracy_score

"""
Load data and setup features and label

Arguments: data_filename which is the file name of formatted data
Return: X as featurs and y as labels of data
"""
def load_data(data_filename, labels):
    print("=== Loading Data ===")
    # Load the data set
    df = pandas.read_csv(data_filename)

    y = df[labels].values
    del df[labels]
    X = df.values
    return X, y

"""
Setting up model parameters based.

Arguments: X as features and y as labels
Return: tupels of gs_cv, model, X as labels for training and testing and
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
    # Create the model
    model = linear_model.LogisticRegression();
    # Parameters to try
    # Apply grid search to find solution
    param_grid = {
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga', 'liblinear']
    }

    print("=== Setting up grid search ===")
    # Run it with four cpus in parallel.
    gs_cv = GridSearchCV(model, param_grid, n_jobs=4, cv=5)
    return gs_cv, model, X_train, X_test, y_train, y_test

"""
Perform grid search

Arguments: Take gs_cv, model, X_train and y_train for training data and X_test
           and y_test for testing data
Return: Nothing
"""
def perform_grid_search(gs_cv, model, X_train, X_test, y_train, y_test):
    print("=== Grid searching in progress ===")
    gs_cv.fit(X_train, y_train)

    print("=== Grid search done ===")
    # Print best result parameters
    best_param = gs_cv.best_params_
    print(best_param)
    """
    Sample output

    {
        'loss': 'huber',
        'learning_rate': 0.1,
        'min_samples_leaf': 9,
        'n_estimators': 3000,
        'max_features': 0.1,
        'max_depth': 6
    }
    """
    # Print Metrics
    print("=" * 30)
    # predictions = gs_cv.predict(X_test)
    # accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy (%): " + str(accuracy))
    mse = mean_absolute_error(y_train, gs_cv.predict(X_train))
    print("Mean abs error Training : %.4f" % mse)
    mse = mean_absolute_error(y_test, gs_cv.predict(X_test))
    print("Mean abs error Training : %.4f" % mse)
    print("=" * 30)
    return best_param

"""
Main function
"""
def main():
    data_filename = 'train_formatted.csv'
    labels = 'Survived'
    X, y = load_data(data_filename, labels)
    gs_cv, model, X_train, X_test, y_train, y_test = setup_data_and_model(X, y)
    best_param = perform_grid_search(gs_cv, model, X_train, X_test, y_train, y_test)
    print("=== Save Best Parametrs ===")
    f = open("grid_search_result.txt","w")
    f.write(str(best_param))
    f.close()
    print("=== Done Everything ===")

if __name__ == '__main__':
    main()
