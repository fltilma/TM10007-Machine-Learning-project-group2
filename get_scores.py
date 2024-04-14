from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
def save_scores_to_csv(train_predictions, validation_predictions, train_labels, validation_labels, best_params, filename):
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_precision = precision_score(train_labels, train_predictions)
    train_recall = recall_score(train_labels, train_predictions)
    train_f1 = f1_score(train_labels, train_predictions)
    validation_accuracy = accuracy_score(validation_labels, validation_predictions)
    validation_precision = precision_score(validation_labels, validation_predictions)
    validation_recall = recall_score(validation_labels, validation_predictions)
    validation_f1 = f1_score(validation_labels, validation_predictions)

    # Write scores to CSV
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Best parameters", best_params])
        writer.writerow(["Metric", "Train", "Validation"])
        writer.writerow(["Accuracy", train_accuracy, validation_accuracy])
        writer.writerow(["Precision", train_precision, validation_precision])
        writer.writerow(["Recall", train_recall, validation_recall])
        writer.writerow(["F1", train_f1, validation_f1])
        

