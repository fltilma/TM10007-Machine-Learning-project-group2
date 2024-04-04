import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.callbacks import EarlyStopping

def load_data():
    """Load data from CSV"""
    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, 'train_data.csv')
    data = pd.read_csv(file_path, index_col=0)
    return data

print("TensorFlow version:", tf.__version__)

def train_neural_network():
    """
    Trains a neural network model using Leave-One-Out cross-validation.

    Args:
    - file_path (str): Path to the CSV file containing the data.

    Returns:
    - trained_model (keras.Model): Trained neural network model.
    """
    # Load the data
    data = load_data()

    # Remove constant features
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]  # Keeps only columns with more than one unique value
    no_constants = data.columns

    # Separate features (X) and target variable (y)
    X = data.drop('label', axis=1)
    y = data['label']

    # Map labels to numerical values
    y = y.map({'benign': 0, 'malignant': 1})

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Define a simple neural network model using TensorFlow's Keras API
    model = keras.Sequential([
        keras.layers.Input(shape=(X_scaled.shape[1],)),  # Input layer
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)),  # Hidden layer with L2 regularization
        keras.layers.Dropout(0.5),  # Dropout layer
        keras.layers.Dense(1, activation='sigmoid')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Binary crossentropy loss
                  metrics=[keras.metrics.Recall()])  # Recall as metrics

    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Convert labels to numpy array
    y_np = np.array(y)

    # Define early stopping
    early_stopping = EarlyStopping(monitor='loss', patience=3, verbose=1)

    # Perform Leave-One-Out cross-validation
    y_pred = np.zeros_like(y_np)
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y_np[train_index], y_np[test_index]

        # Train the model on the training data
        model.fit(X_train, y_train, epochs=30, verbose=1, callbacks=[early_stopping])  # Fit the model with Leave-One-Out Early Stopping

        # Predict on the test data
        y_pred[test_index] = (model.predict(X_test) > 0.5).astype(int)

    return model, y_pred, y_np, no_constants

if __name__ == "__main__":
    model, y_pred, y_np, no_constants = train_neural_network()
    accuracy = accuracy_score(y_np, y_pred)
    precision = precision_score(y_np, y_pred)
    recall = recall_score(y_np, y_pred)
    f1 = f1_score(y_np, y_pred)
    roc_auc = roc_auc_score(y_np, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print("Classifier training complete.")