import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, make_scorer, f1_score, cohen_kappa_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate

from scipy.stats import spearmanr, pearsonr


def get_ratio(y_train_cv):
    return float(y_train_cv.value_counts()[0]) / y_train_cv.value_counts()[1]


def cross_validation_generator(X, y, fold, random_state=42):
    skf = StratifiedKFold(n_splits=fold, random_state=random_state, shuffle=True)
    return skf.split(X, y)


def show_ConfusionMatrix_folds(avg_confusion_matrix):
    # Plot the averaged confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=avg_confusion_matrix)
    disp.plot(cmap=plt.cm.Blues, values_format='.1f')
    plt.title('Averaged Confusion Matrix')


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
    # print("Confusion Matrix (Test):")
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
    plt.figure(figsize=(14, 8))
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


def perform_cross_validation(model, X_train, y_train, n_splits=5, random_state=42):
    # Initialize StratifiedKFold with the given number of splits and random state
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Define the scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': make_scorer(balanced_accuracy_score),
        'roc_auc': 'roc_auc',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'kappa': make_scorer(cohen_kappa_score)
    }

    # Perform cross-validation
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
    
    # Print the cross-validation results
    print(f"Cross-Validation Accuracy Scores: {cv_results['test_accuracy']}")
    print(f"Mean Cross-Validation Accuracy: {round(cv_results['test_accuracy'].mean(), 2)}")
    
    print(f"Cross-Validation Balanced Accuracy Scores: {cv_results['test_balanced_accuracy']}")
    print(f"Mean Cross-Validation Balanced Accuracy: {round(cv_results['test_balanced_accuracy'].mean(), 2)}")
    
    print(f"Cross-Validation AUC Scores: {cv_results['test_roc_auc']}")
    print(f"Mean Cross-Validation AUC: {round(cv_results['test_roc_auc'].mean(), 2)}")
    
    print(f"Cross-Validation Precision Scores: {cv_results['test_precision']}")
    print(f"Mean Cross-Validation Precision: {round(cv_results['test_precision'].mean(), 2)}")
    
    print(f"Cross-Validation Recall Scores: {cv_results['test_recall']}")
    print(f"Mean Cross-Validation Recall: {round(cv_results['test_recall'].mean(), 2)}")
    
    print(f"Cross-Validation F1 Scores: {cv_results['test_f1']}")
    print(f"Mean Cross-Validation F1: {round(cv_results['test_f1'].mean(), 2)}")
    
    print(f"Cross-Validation Kappa Scores: {cv_results['test_kappa']}")
    print(f"Mean Cross-Validation Kappa: {round(cv_results['test_kappa'].mean(), 2)}")
    
    return cv_results


def plot_prediction_distributions(X_test, y_test, y_pred, feature_1='relationships', feature_2='founded_at_year'):
    """
    Plots the distribution of specified features for correct and incorrect predictions.

    Parameters:
    - X_test: DataFrame containing the test features.
    - y_test: Actual target values for the test set.
    - y_pred: Predicted target values.
    - feature_1: The name of the first feature to plot (default is 'relationships').
    - feature_2: The name of the second feature to plot (default is 'founded_at_year').
    """
    
    # Create a DataFrame to store results
    results_df = X_test.copy()
    results_df['actual'] = y_test
    results_df['predicted'] = y_pred
    results_df['correct'] = (y_pred == y_test)

    # Separate correct and incorrect predictions
    correct_predictions = results_df[results_df['correct'] == True]
    incorrect_predictions = results_df[results_df['correct'] == False]

    plt.figure(figsize=(14, 5))

    # Plot distribution for the first feature on the left
    plt.subplot(1, 2, 1)
    sns.kdeplot(correct_predictions[feature_1], shade=True, label='Correct', color='g')
    sns.kdeplot(incorrect_predictions[feature_1], shade=True, label='Incorrect', color='r')
    plt.title(f'Distribution of "{feature_1}"')
    plt.legend()

    # Plot distribution for the second feature on the right
    plt.subplot(1, 2, 2)
    sns.kdeplot(correct_predictions[feature_2], shade=True, label='Correct', color='g')
    sns.kdeplot(incorrect_predictions[feature_2], shade=True, label='Incorrect', color='r')
    plt.title(f'Distribution of "{feature_2}"')
    plt.legend()

    # Add a big title to the entire figure
    plt.suptitle('High Effect Feature Distributions for Correct and Incorrect Predictions', fontsize=16, y=1.05)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()


def plot_target_distribution(fold_targets_df):
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Plot the distribution for 'Acquired'
    acquired_data = fold_targets_df[fold_targets_df['Target'] == 1]
    acquired_counts = acquired_data['Fold'].value_counts().sort_index()
    axes[0].bar(acquired_counts.index, acquired_counts.values, color='skyblue')
    axes[0].set_title('Distribution of Acquired Targets Across Folds')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Count')

    # Plot the distribution for 'Not Acquired'
    not_acquired_data = fold_targets_df[fold_targets_df['Target'] == 0]
    not_acquired_counts = not_acquired_data['Fold'].value_counts().sort_index()
    axes[1].bar(not_acquired_counts.index, not_acquired_counts.values, color='salmon')
    axes[1].set_title('Distribution of Not Acquired Targets Across Folds')
    axes[1].set_xlabel('Fold')

    plt.suptitle('Target Distribution Across Folds')
    
    
def plot_feature_importances_kfold_agg(all_feature_importances, feature_names, n_features=10):
    # Calculate average feature importances
    avg_importances = np.mean(all_feature_importances, axis=0)

    # Create a DataFrame for easy sorting and selection
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    # Select the top n_features
    top_importances = importances_df.head(n_features)

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(top_importances['Feature'], top_importances['Importance'], color='skyblue')
    plt.title(f'Top {n_features} Feature Importances Across Folds')
    plt.xlabel('Average Importance')
    plt.ylabel('Feature')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important features on top
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