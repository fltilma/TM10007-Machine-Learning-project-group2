from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocessing
from ploty_learning_curve import ploty_learning_curve
from get_scores import save_scores_to_csv

# Load data
X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessing()
print('Data loading successful')

# Convert labels to numpy arrays
y_train = np.array(y_train).astype(int)
y_validation = np.array(y_validation).astype(int)

# Initialize pipelines for SVM and KNN with PCA for feature selection
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Use PCA for feature selection
    ('svm', SVC())
])

knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),  # Use PCA for feature selection
    ('knn', KNeighborsClassifier())
])

# Define parameter grids for SVM and KNN
svm_param_grid = {
    'pca__n_components': [20],  # Adjust the number of components as needed
    'svm__kernel': ['linear'],
    'svm__C': [3]
}

knn_param_grid = {
    'pca__n_components': [12],  # Adjust the number of components as needed
    'knn__n_neighbors': [5]
}

scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall',
           'f1' : 'f1'
}

# Instantiate GridSearchCV with pipelines for SVM and KNN
svm_grid_search = GridSearchCV(svm_pipeline, svm_param_grid, scoring=scoring, refit='f1', cv=LeaveOneOut())
knn_grid_search = GridSearchCV(knn_pipeline, knn_param_grid, scoring=scoring, refit='f1', cv=LeaveOneOut())

# Fit GridSearchCV for SVM and KNN
svm_grid_search.fit(X_train, y_train)
knn_grid_search.fit(X_train, y_train)

# Get best models
best_svm_model = svm_grid_search.best_estimator_
svm_params = svm_grid_search.best_params_
best_knn_model = knn_grid_search.best_estimator_
knn_params = knn_grid_search.best_params_

# Predictions from base models on validation set
svm_pred_train = best_svm_model.predict(X_train)
svm_pred_validation = best_svm_model.predict(X_validation)
svm_pred_test = best_svm_model.predict(X_test)
knn_pred_train = best_knn_model.predict(X_train)
knn_pred_validation = best_knn_model.predict(X_validation)
knn_pred_test = best_knn_model.predict(X_test)


save_scores_to_csv(svm_pred_train, svm_pred_validation, y_train, y_validation, svm_params, "zsvm_scores.csv")
save_scores_to_csv(svm_pred_train, svm_pred_test, y_train, y_test, svm_params, "zsvm_test_scores.csv")
save_scores_to_csv(knn_pred_train, knn_pred_validation, y_train, y_validation, knn_params, "zknn_scores.csv")
save_scores_to_csv(knn_pred_train, knn_pred_test, y_train, y_test, knn_params, "zknn_test_scores.csv")
# Define a range of thresholds
thresholds = np.linspace(0.1, 0.9, 9)

# Initialize variables to store best threshold and corresponding metric
best_threshold = None
best_metric = -1  # Initialize to a low value

# Iterate over each threshold
for threshold in thresholds:
    # Convert predictions to binary using the current threshold
    svm_pred_validation_binary = (svm_pred_validation > threshold).astype(int)
    knn_pred_validation_binary = (knn_pred_validation > threshold).astype(int)

    # Prepare meta-features for validation set
    meta_features_validation = np.column_stack((svm_pred_validation_binary, knn_pred_validation_binary))

    # Meta-model with regularization
    meta_model = LogisticRegression(penalty='l2', C=1.0)  # Regularization

    # Fit meta-model on meta-features
    meta_model.fit(meta_features_validation, y_validation)

    # Predictions from meta-model
    stacking_validation_predictions = meta_model.predict(meta_features_validation)

    # Calculate metric of interest (e.g., F1 score)
    metric = f1_score(y_validation, stacking_validation_predictions)

    # Update best threshold and corresponding metric if current metric is better
    if metric > best_metric:
        best_metric = metric
        best_threshold = threshold

# Use the best threshold for predictions
print("Best Threshold:", best_threshold)

# Convert predictions to binary using the best threshold for both training and validation sets
svm_pred_train_binary = (best_svm_model.predict(X_train) > best_threshold).astype(int)
knn_pred_train_binary = (best_knn_model.predict(X_train) > best_threshold).astype(int)
meta_features_train = np.column_stack((svm_pred_train_binary, knn_pred_train_binary))

# Meta-model for training set with regularization
meta_model_train = LogisticRegression(penalty='l2', C=1.0)  # You can adjust the regularization parameter C as needed

# Fit meta-model on meta-features for training set
meta_model_train.fit(meta_features_train, y_train)

# Predictions from meta-model on training set
stacking_train_predictions = meta_model_train.predict(meta_features_train)

# Evaluation on training set
stacking_train_recall = recall_score(y_train, stacking_train_predictions)
stacking_train_f1 = f1_score(y_train, stacking_train_predictions)
stacking_train_accuracy = accuracy_score(y_train, stacking_train_predictions)
stacking_train_precision = precision_score(y_train, stacking_train_predictions)

# Print scores for training set
print("Training Set Scores:")
print("Recall:", stacking_train_recall)
print("F1 Score:", stacking_train_f1)
print("Accuracy:", stacking_train_accuracy)
print("Precision:", stacking_train_precision)

# Convert predictions to binary using the best threshold for the validation set
svm_pred_validation_binary = (best_svm_model.predict(X_validation) > best_threshold).astype(int)
knn_pred_validation_binary = (best_knn_model.predict(X_validation) > best_threshold).astype(int)
meta_features_validation = np.column_stack((svm_pred_validation_binary, knn_pred_validation_binary))

# Predictions from meta-model on validation set
stacking_validation_predictions = meta_model_train.predict(meta_features_validation)

# Evaluation on validation set
stacking_validation_recall = recall_score(y_validation, stacking_validation_predictions)
stacking_validation_f1 = f1_score(y_validation, stacking_validation_predictions)
stacking_validation_accuracy = accuracy_score(y_validation, stacking_validation_predictions)
stacking_validation_precision = precision_score(y_validation, stacking_validation_predictions)

# Print scores for validation set
print("Validation Set Scores:")
print("Recall:", stacking_validation_recall)
print("F1 Score:", stacking_validation_f1)
print("Accuracy:", stacking_validation_accuracy)
print("Precision:", stacking_validation_precision)

# Print the number of components selected by PCA in the SVM pipeline
print("Number of components selected by PCA in the SVM pipeline:", best_svm_model.named_steps['pca'].n_components_)

# Print the number of components selected by PCA in the KNN pipeline
print("Number of components selected by PCA in the KNN pipeline:", best_knn_model.named_steps['pca'].n_components_)

# Plot learning curve for SVM
title = "Learning Curves (SVM)"
ploty_learning_curve(best_svm_model, title, X_train, y_train, cv=LeaveOneOut(), n_jobs=-1)

# Plot learning curve for KNN
title = "Learning Curves (KNN)"
ploty_learning_curve(best_knn_model, title, X_train, y_train, cv=LeaveOneOut(), n_jobs=-1)

plt.show()


# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, svm_pred_test)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of linear SVM classifier on test data')
plt.legend(loc="lower right")
plt.show()

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_validation, knn_pred_validation)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of kNN classifier on validation data')
plt.legend(loc="lower right")
plt.show()

# Compute ROC curve and ROC area for each class
fpr, tpr, _ = roc_curve(y_test, knn_pred_test)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve of kNN classifier on test data')
plt.legend(loc="lower right")
plt.show()