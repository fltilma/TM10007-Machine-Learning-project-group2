'''This module loads the data and prepares it for feature selection'''
import os
import pandas as pd
def preprocessing():
    print('Data loading initialized')
    def load_data(csv):
        """Load data from CSV"""
        # Get the directory path where the script is located
        this_directory = os.path.dirname(os.path.abspath(__file__))

        # Assuming 'Liver_radiomicFeatures.csv' is the name of your CSV file
        file_path = os.path.join(this_directory, csv)

        # Read the CSV file into a DataFrame
        data = pd.read_csv(file_path, index_col=0)  
        return data

    # Load the data
    test_data = load_data('test_data.csv')
    train_data = load_data('train_data.csv')
    validation_data = load_data('validation_data.csv')

    X_train = train_data.drop('label', axis=1)
    X_test = test_data.drop('label', axis=1)
    X_validation = validation_data.drop('label', axis=1)
    y_train = train_data['label']
    y_test = test_data['label']
    y_validation = validation_data['label']

    # Map labels to numerical values to make it easier for the model
    y_train = y_train.map({'benign': 0, 'malignant': 1})
    y_test = y_test.map({'benign': 0, 'malignant': 1})
    y_validation = y_validation.map({'benign': 0, 'malignant': 1})
    return(X_train, X_test, X_validation, y_train, y_test, y_validation)