import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score

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

# Step 2: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Step 4: Train a classifier using the PCA-transformed features
classifier = SVC(kernel='rbf')  # Example: RBF kernel
classifier.fit(X_train_pca, y_train)

# Step 5: Predict on the test set
y_pred = classifier.predict(X_test_pca)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Test Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")