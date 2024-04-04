import os
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data():
    """Load data from CSV"""
    this_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(this_directory, 'train_data.csv')
    data = pd.read_csv(file_path, index_col=0)
    return data

def train_classifier():
    # Load the data
    data = load_data()

    # Remove constant features
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]
    no_constants = data.columns

    # Separate features (X) and target variable (y)
    X = data.drop('label', axis=1)
    y = data['label']
    y = y.map({'benign': 0, 'malignant': 1})

    # Initialize Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=1)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train the classifier
    rf_classifier.fit(X_scaled, y)

    # Get feature importances
    feature_importances = rf_classifier.feature_importances_
    sorted_indices = feature_importances.argsort()[::-1]
    selected_feature_indices = sorted_indices[:50]
    X_selected = X_scaled[:, selected_feature_indices]

    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Initialize SVC classifier with hyperparameter tuning
    svc_classifier = SVC(class_weight='balanced')
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}
    grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='recall')
    grid_search.fit(X_selected, y)
    best_svc = grid_search.best_estimator_

    y_pred = cross_val_predict(best_svc, X_selected, y, cv=loo)

    return best_svc, scaler, selected_feature_indices, y_pred, y, no_constants

if __name__ == "__main__":
    best_svc, scaler, selected_feature_indices, y_pred, y, no_constants = train_classifier()
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print("Classifier training complete.")
