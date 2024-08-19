from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np


def show_ConfusionMatrix_test(y_test, y_test_pred, test_confusion_matrix_title = "Confusion Matrix (Test)"):
    """
    Displays the confusion matrix for the test set predictions.

    Parameters:
    - y_test: array-like, true labels for the test set.
    - y_test_pred: array-like, predicted labels for the test set.
    - test_confusion_matrix_title: str, title for the confusion matrix plot (default is "Confusion Matrix (Test)").
    
    This function prints the confusion matrix to the console and then 
    visualizes it using a heatmap with the specified title.
    """
    # Compute the confusion matrix
    conf_matrix_log_reg = confusion_matrix(y_test, y_test_pred)
    
    # Print the confusion matrix
    print("Confusion Matrix (Test):")
    print(conf_matrix_log_reg)
    
    # Plot the confusion matrix using ConfusionMatrixDisplay
    ConfusionMatrixDisplay(conf_matrix_log_reg).plot()
    plt.title(test_confusion_matrix_title)
    plt.show()
    
    
def get_precision_and_recall(y, y_pred):
    """
    Calculates and returns the precision and recall for the given predictions.

    Parameters:
    - y: array-like, true labels.
    - y_pred: array-like, predicted labels.

    Returns:
    - tuple: precision and recall values rounded to 4 decimal places.
    
    This function computes precision and recall, which are metrics used to evaluate 
    the performance of classification models. Precision is the ratio of correctly predicted 
    positive observations to the total predicted positives, while recall is the ratio of 
    correctly predicted positive observations to all observations in the actual class.
    """
    return round(precision_score(y, y_pred), 4), round(recall_score(y, y_pred), 4)


def plot_feature_importances(model, feature_names, num_of_features=10):
    """
    Plot the top feature importances for a given model.

    Parameters:
    - model: Trained model with feature_importances_ attribute (e.g., XGBClassifier).
    - feature_names: List or array of feature names corresponding to model features.
    - num_of_features: Number of top features to plot (default is 10).
    """
    # Extract feature importances
    importance = model.feature_importances_

    # Create a DataFrame for feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(num_of_features)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Top {num_of_features} Feature Importances')
    plt.gca().invert_yaxis()
    plt.show()


def plot_auc_roc_curve(y_true, y_prob, model_name='Model'):
    """
    Plot the AUC-ROC curve for a given model.

    Parameters:
    - y_true: Array-like, true labels of the test set.
    - y_prob: Array-like, predicted probabilities of the positive class.
    - model_name: String, name of the model for the plot label (default is 'Model').
    """
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_roc = auc(fpr, tpr)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} AUC-ROC (area = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()



# def find_optimal_threshold(target, predicted):
#     """ Find the optimal probability cutoff point for a classification model related to event rate
#     Parameters:
#     target : Matrix with dependent or target data, where rows are observations
#     predicted : Matrix with predicted data, where rows are observations

#     Returns:
#     list type, with optimal cutoff value
#     """
#     fpr, tpr, threshold = roc_curve(target, predicted)
#     i = np.arange(len(tpr))
#     roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
#     roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
#     return list(roc_t['threshold'])