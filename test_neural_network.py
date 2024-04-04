import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from neural_function import train_neural_network

def load_validation_data():
    """Load validation data from CSV"""
    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, 'validation_data.csv')
    validation_data = pd.read_csv(file_path, index_col=0)
    return validation_data

# Load the trained model
trained_model, y_pred, y_train, no_constants = train_neural_network()

# Load validation data
validation_data = load_validation_data()
validation_data = validation_data[no_constants]

# Separate features (X) and target variable (y)
X_val = validation_data.drop('label', axis=1)
y_val = validation_data['label'].map({'benign': 0, 'malignant': 1})

# Feature Scaling on validation data
scaler = StandardScaler()
X_val_scaled = scaler.fit_transform(X_val)

# Predict on validation data
y_pred_val = (trained_model.predict(X_val_scaled) > 0.5).astype(int)

# Calculate evaluation metrics on validation data
accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
roc_auc = roc_auc_score(y_val, y_pred_val)

# Print the evaluation metrics on validation data
print("\nPerformance on validation data:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"ROC-AUC: {roc_auc}")
