import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from classifier_function import train_classifier

def load_validation_data():
    """Load validation data from CSV"""
    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, 'validation_data.csv')
    validation_data = pd.read_csv(file_path, index_col=0)
    return validation_data

if __name__ == "__main__":
    best_svc, scaler, selected_feature_indices, y_pred, y, no_constants = train_classifier()
    # Load validation data
    validation_data = load_validation_data()

    # Preprocess validation data (remove constant features and select the same features as training data)
    validation_data = validation_data[no_constants]
    X_val = validation_data.drop('label', axis=1)
    y_val = validation_data['label'].map({'benign': 0, 'malignant': 1})

    # Scale features using the same scaler used during training
    X_val_scaled = scaler.transform(X_val)

    # Select the same top 50 features used during training
    X_val_selected = X_val_scaled[:, selected_feature_indices]

    # Make predictions on validation data
    y_pred_val = best_svc.predict(X_val_selected)

    # Evaluate performance
    accuracy_val = accuracy_score(y_val, y_pred_val)
    precision_val = precision_score(y_val, y_pred_val)
    recall_val = recall_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val)
    roc_auc_val = roc_auc_score(y_val, y_pred_val)

    print("Performance on Validation Data:")
    print(f"Accuracy: {accuracy_val}")
    print(f"Precision: {precision_val}")
    print(f"Recall: {recall_val}")
    print(f"F1-score: {f1_val}")
    print(f"ROC-AUC: {roc_auc_val}")