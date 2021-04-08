from sklearn.metrics import multilabel_confusion_matrix
import numpy as np
import pandas as pd
import pickle

# Helper methods for evaluating a CNN using data in the format produced by the model class.

def get_evaluation_dataframe(training_file):

    cnn_training_results = pickle.load(open(training_file, "rb"))

    eval_list = []
    for epoch in range(len(cnn_training_results["epoch_acc"])):
        epoch_list = [cnn_training_results["epoch_acc"][epoch]]
        precisions = []
        recalls = []

        y_pred = cnn_training_results["predicted_labels"][epoch]
        y_true = cnn_training_results["ground_truth_labels"][epoch]

        confusion_matrices = multilabel_confusion_matrix(y_true, y_pred)
        for matrix in confusion_matrices:
            precision = matrix[1][1] / (matrix[1][0] + matrix[1][1])
            recall = matrix[1][1] / (matrix[0][1] + matrix[1][1])

            precisions.append(precision)
            recalls.append(recall)

        epoch_list.extend(precisions + recalls)
        eval_list.append(epoch_list)


        # Confusion matrices contains one matrix per label, so its length is the number of labels.
        dataframe_headings = ["accuracy"]
        for label in range(len(confusion_matrices)):
            dataframe_headings.append("precision_" + str(label))
        for label in range(len(confusion_matrices)):
            dataframe_headings.append("recall_" + str(label))

    return pd.DataFrame(eval_list, columns=dataframe_headings)