import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocessing

X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessing()
print('Data loading succesfull')
#initializing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('svm', SVC())
])

#initialize grid
param_grid = {
    'pca__n_components': range(1,50),
    'svm__kernel': ['linear'],
    'svm__C': [1]
}

scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall',
           'f1' : 'f1'
}

# Instantiate GridSearchCV with pipeline
print("Searching grid...")
grid_search = GridSearchCV(pipeline, param_grid, scoring=scoring, refit='f1', cv=LeaveOneOut())

# Fit GridSearchCV
grid_search.fit(X_train, y_train)
print("Model optimized")
# Evaluate best model
best_model = grid_search.best_estimator_

# Evaluate best model on test data
y_pred = best_model.predict(X_validation)
y_pred_train = best_model.predict(X_train)

train_accuracy = accuracy_score(y_train, y_pred_train)
train_precision = precision_score(y_train, y_pred_train)
train_recall = recall_score(y_train, y_pred_train)
train_f1 = f1_score(y_train, y_pred_train)
validation_accuracy = accuracy_score(y_validation, y_pred)
validation_precision = precision_score(y_validation, y_pred)
validation_recall = recall_score(y_validation, y_pred)
validation_f1 = f1_score(y_validation, y_pred)
print("Best parameters:", grid_search.best_params_)
print("Train accuracy:", train_accuracy, "Train precision:", train_precision, "Train recall:", train_recall, 'Train f1 score', train_f1)
print("Validation accuracy:", validation_accuracy, "Validation precision:", validation_precision, "Validation recall:", validation_recall, 'Validation f1 score', validation_f1)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot learning curve for SVM
title = "Learning Curves (SVM)"
plot_learning_curve(best_model, title, X_train, y_train, cv=LeaveOneOut(), n_jobs=-1)
plt.show()