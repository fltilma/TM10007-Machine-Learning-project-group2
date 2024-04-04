import os
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.callbacks import EarlyStopping

print("TensorFlow version:", tf.__version__)

def load_data():
    """Load data from CSV"""
    # Get the directory path where the script is located
    this_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Assuming 'train_data.csv' is the name of your CSV file
    file_path = os.path.join(this_directory, 'train_data.csv')
    
    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, index_col=0)
    
    return data

# Load the data
data = load_data()

# Remove constant features
data = data.loc[:, data.apply(pd.Series.nunique) != 1]  # Keeps only columns with more than one unique value

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
    keras.layers.Input(shape=(X_scaled.shape[1],)), # Input layer
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)), # Hidden layer with L2 regularization
    keras.layers.Dropout(0.5), # Dropout layer
    keras.layers.Dense(1, activation='sigmoid') # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy', # Binary crossentropy loss
              metrics=[keras.metrics.Recall()]) # recall as metrics

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
    model.fit(X_train, y_train, epochs=30, verbose=1, callbacks=[early_stopping]) # Fit the model with Leave-One-Out Early Stopping

    # Predict on the test data
    y_pred[test_index] = (model.predict(X_test) > 0.5).astype(int)

# Calculate evaluation metrics
accuracy = accuracy_score(y_np, y_pred)
precision = precision_score(y_np, y_pred)
recall = recall_score(y_np, y_pred)
f1 = f1_score(y_np, y_pred)
roc_auc = roc_auc_score(y_np, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"ROC-AUC: {roc_auc}")
