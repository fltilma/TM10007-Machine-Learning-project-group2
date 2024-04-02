"""Function to determine wheter tissue is benign or malignant""" 
import os
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data():
    """Load data from CSV"""
    # Get the directory path where the script is located
    this_directory = os.path.dirname(os.path.abspath(__file__))

    # Assuming 'Liver_radiomicFeatures.csv' is the name of your CSV file
    file_path = os.path.join(this_directory, 'train_data.csv')

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path, index_col=0)

    return data

# Load the data
data = load_data()

# Remove constant features by keeping columns with more than one unique value
data = data.loc[:, data.apply(pd.Series.nunique) != 1]  

# Separate features (X) and target variable (y)
# Drops the label column at the X-axis
# Looks at the data of the 'label' at the Y-axis 
X = data.drop('label', axis=1)
y = data['label']

# Map labels to numerical values to make it easier for the model
y = y.map({'benign': 0, 'malignant': 1})

# Initialize Random Forest classifier, needed for scaling and feature selection
rf_classifier = RandomForestClassifier(random_state=1)

# Feature Scaling to make sure features contribute equally
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the classifier on the scaled data
rf_classifier.fit(X_scaled, y)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Sort features by importance in descending order
sorted_indices = feature_importances.argsort()[::-1]

# Select the top 50 features
selected_feature_indices = sorted_indices[:50]

# Select the top 50 features
X_selected = X_scaled[:, selected_feature_indices]

# Initialize Leave-One-Out cross-validator
loo = LeaveOneOut()

# Initialize Support Vector Classifier (SVC) with hyperparameter tuning
svc_classifier = SVC(class_weight='balanced')

# Define the hyperparameters grid
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

# Perform grid search cross-validation
grid_search = GridSearchCV(svc_classifier, param_grid, cv=5, scoring='recall')
grid_search.fit(X_selected, y)

# Get the best estimator
best_svc = grid_search.best_estimator_

# Perform LOOCV on the best SVC classifier
y_pred = cross_val_predict(best_svc, X_selected, y, cv=loo)

# Calculate evaluation metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)
roc_auc = roc_auc_score(y, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"ROC-AUC: {roc_auc}")
