import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocessing

# Load data
X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessing()
print('Data loading successful')

# Define range of number of features
num_features_range = range(1, 105)

# Initialize lists to store performance metrics
train_accuracies = []
train_precisions = []
train_recalls = []
train_f1_scores = []
validation_accuracies = []
validation_precisions = []
validation_recalls = []
validation_f1_scores = []

# Iterate through different number of features
for num_features in num_features_range:
    # Initialize pipeline with SelectKBest
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(f_classif, k=num_features)),
        ('svm', SVC(kernel='linear'))
    ])

    # Fit pipeline on training data
    pipeline.fit(X_train, y_train)

    # Predictions on training and validation sets
    y_pred_train = pipeline.predict(X_train)
    y_pred_validation = pipeline.predict(X_validation)

    # Calculate performance metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)
    validation_accuracy = accuracy_score(y_validation, y_pred_validation)
    validation_precision = precision_score(y_validation, y_pred_validation)
    validation_recall = recall_score(y_validation, y_pred_validation)
    validation_f1 = f1_score(y_validation, y_pred_validation)

    # Append metrics to lists
    train_accuracies.append(train_accuracy)
    train_precisions.append(train_precision)
    train_recalls.append(train_recall)
    train_f1_scores.append(train_f1)
    validation_accuracies.append(validation_accuracy)
    validation_precisions.append(validation_precision)
    validation_recalls.append(validation_recall)
    validation_f1_scores.append(validation_f1)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(num_features_range, train_accuracies, label='Train Accuracy')
plt.plot(num_features_range, train_precisions, label='Train Precision')
plt.plot(num_features_range, train_recalls, label='Train Recall')
plt.plot(num_features_range, train_f1_scores, label='Train F1 Score')
plt.plot(num_features_range, validation_accuracies, label='Validation Accuracy')
plt.plot(num_features_range, validation_precisions, label='Validation Precision')
plt.plot(num_features_range, validation_recalls, label='Validation Recall')
plt.plot(num_features_range, validation_f1_scores, label='Validation F1 Score')

plt.xlabel('Number of Features')
plt.ylabel('Performance Metrics')
plt.title('Performance Metrics vs Number of Features')
plt.legend()
plt.grid(True)
plt.xticks(np.arange(0, 105, step=5))
plt.tight_layout()
plt.show()
