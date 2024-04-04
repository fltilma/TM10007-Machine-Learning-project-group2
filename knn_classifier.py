"""Function to determine wheter tissue is benign or malignant""" 
import os
import pandas as pd
from sklearn.model_selection import LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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

# Separate features (X) and target variable (y)
# Drops the label column at the X-axis
# Looks at the data of the 'label' at the Y-axis
X_train = train_data.drop('label', axis=1)
X_test = test_data.drop('label', axis=1)
y_train = train_data['label']
y_test = test_data['label']

# Map labels to numerical values to make it easier for the model
y_train = y_train.map({'benign': 0, 'malignant': 1})
y_test = y_test.map({'benign': 0, 'malignant': 1})

# Initialize Random Forest classifier, needed for scaling and feature selection
rf_classifier = RandomForestClassifier(random_state=1)

# Feature Scaling to make sure features contribute equally
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the classifier on the scaled data
rf_classifier.fit(X_train_scaled, y_train)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Sort features by importance in descending order
sorted_indices = feature_importances.argsort()[::-1]

accuracy_values = []
test_accuracy_values = []
precision_values = []
test_precision_values = []

for i in range(20, 40):
    print(f"Checking top {i} best features...")
    # Select the top 101 features
    selected_feature_indices = sorted_indices[:i]

    # Select the top x features
    X_selected = X_train_scaled[:, selected_feature_indices]
    X_test_selected = X_test_scaled[:, selected_feature_indices]

    # Initialize Leave-One-Out cross-validator
    loo = LeaveOneOut()

    # Initialize k-Nearest Neighbors (k-NN) classifier
    knn_classifier = KNeighborsClassifier()

    # Define the hyperparameters grid
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

    # Perform grid search cross-validation
    grid_search = GridSearchCV(knn_classifier, param_grid, cv=5, scoring='precision')
    grid_search.fit(X_selected, y_train)

    # Get the best estimator
    best_knn = grid_search.best_estimator_

    # Perform LOOCV on the best SVC classifier
    y_pred = cross_val_predict(best_knn, X_selected, y_train, cv=loo)
    y_test_pred = best_knn.predict(X_test_selected)
    print(y_test_pred)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred)
    #recall = recall_score(y_train, y_pred)
    #f1 = f1_score(y_train, y_pred)
    #roc_auc = roc_auc_score(y_train, y_pred)

    #Calculate values from test set
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)

    # Print the evaluation metrics
    #print(f"Accuracy: {accuracy}")
    #print(f"Precision: {precision}")
    #print(f"Recall: {recall}")
    #print(f"F1-score: {f1}")
    #print(f"ROC-AUC: {roc_auc}")
    accuracy_values.append(accuracy)
    test_accuracy_values.append(test_accuracy)
    precision_values.append(precision)
    test_precision_values.append(test_precision)

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Plot training and test accuracies
ax1.plot(range(20, 40), accuracy_values, marker='o', color='blue', label='Train Accuracy')
ax1.plot(range(20, 40), test_accuracy_values, marker='o', color='red', label='Test Accuracy')
ax1.set_xlabel('Number of Selected Features (i)')
ax1.set_ylabel('Accuracy')
ax1.set_title('Accuracy vs Number of Selected Features')
ax1.legend()
ax1.grid(True)

# Plot training and test precisions
ax2.plot(range(20, 40), precision_values, marker='o', color='blue', label='Train Precision')
ax2.plot(range(20, 40), test_precision_values, marker='o', color='red', label='Test Precision')
ax2.set_xlabel('Number of Selected Features (i)')
ax2.set_ylabel('Precision')
ax2.set_title('Precision vs Number of Selected Features')
ax2.legend()
ax2.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()