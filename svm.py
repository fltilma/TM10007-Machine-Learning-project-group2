from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from preprocessing import preprocessing

X_train, X_test, X_validation, y_train, y_test, y_validation = preprocessing()
print('Data loading succesfull')
#initializing pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest()),
    ('svm', SVC())
])

#initialize grid
param_grid = {
    'selector__score_func':[f_classif],
    'selector__k': range(20,110,2),
    'svm__kernel': ['linear']
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
