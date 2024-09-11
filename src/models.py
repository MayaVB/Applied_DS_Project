import numpy as np
from xgboost import XGBClassifier
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, roc_auc_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.stats import mode

from eval import show_ConfusionMatrix_test, show_ConfusionMatrix_folds, plot_target_distribution
from getdata import add_nasdaq_annual_changes, add_economic_indicators
from preprocess import preprocess_data_classifier
from utils import load_data

def train_rfv2_model(X_train, y_train):
    # Define hyperparameters for tuning
    param_grid = {
        'n_estimators': [150, 200, 250],
        'max_depth': [35, 40, 45],
        'min_samples_split': [2, 3],
        'min_samples_leaf': [2, 4],
        'bootstrap': [True, False]
    }
    
    # Initialize the Random Forest model
    rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='roc_auc', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_rf_model = grid_search.best_estimator_
    
    return best_rf_model


def train_xgb_model(X_train, y_train):
    """Train an XGBoost classifier and return the trained model."""
    xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb_clf.fit(X_train, y_train)
    return xgb_clf

def train_rf_model(X_train, y_train):
    """Train a RandomForest classifier and return the trained model."""
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train, y_train)
    return rf_clf

def train_svm_model(X_train, y_train):
    """Train a Support Vector Machine classifier and return the trained model."""
    svm_clf = SVC(probability=True)
    svm_clf.fit(X_train, y_train)
    return svm_clf


def train_decision_tree_model(X_train, y_train):
    """Train a Decision Tree classifier and return the trained model."""
    dt_clf = DecisionTreeClassifier(random_state=42)
    dt_clf.fit(X_train, y_train)
    return dt_clf


def predict_model(model, X_test):
    """Make predictions and estimate probabilities."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

def predict_with_ensemble_model(trained_models, X_test):
    preds = []
    probs = []
    for i in range(len(trained_models)):
        # Predict on the test set
        pred, prob = predict_model(trained_models[i], X_test)
        preds.append(pred)
        probs.append(prob)
    predictions = np.column_stack(preds)

    # Majority voting
    ensemble_pred, _ = mode(predictions, axis=1)
    ensemble_pred = ensemble_pred.ravel()

    # Combine probabilities (e.g., by averaging them)
    ensemble_prob = np.array(probs).mean(axis=0)
    return ensemble_pred, ensemble_prob

def train_and_calc_ensemble_metrics(models, X_train, y_train, X_test, random_state):
    # Set a fixed random state for reproducibility
    for model in models:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=random_state)  # Ensure random_state is fixed for all models
        model.fit(X_train, y_train)
    return predict_with_ensemble_model(models, X_test)

def cross_validate_ensemble_using_StratifiedKFold_short(models_for_ensemble, X, y, n_splits, th, 
                                                  random_state, show_auc_curve = True):
  skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)   
  
  metrics_arr = {
        'Threshold': [],
        'AUC-ROC': [],
        'Accuracy': [],
        'Balanced Accuracy': [],
        'Precision': [],
        'Recall': []
      }  

  for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
      y_test_cv = y.iloc[val_index]
      ensemble_pred, ensemble_prob = train_and_calc_ensemble_metrics(models_for_ensemble, X.iloc[train_index], y.iloc[train_index], X.iloc[val_index], random_state)
      res = evaluate_model(y_test_cv, ensemble_pred, ensemble_prob, threshold=th) 
      for key, value in res.items():
        metrics_arr[key].append(value)
      if show_auc_curve:
        plot_auc_roc_curve(y_test_cv, ensemble_prob, model_name='Ensemble')
  k = list(metrics_arr.keys())[1:]
  v_mean = [round(np.mean(metrics_arr[key]), 2) for key in metrics_arr][1:]
  v_std = [round(np.std(metrics_arr[key]), 2) for key in metrics_arr][1:]
  means_dict = dict(zip(k, v_mean))
  std_dict = dict(zip(k, v_std))
  print(means_dict)
  return metrics_arr, means_dict, std_dict
                                                      
def cross_validate_ensemble_using_StratifiedKFold(models, X, y, n_splits=5, random_state=None, print_avg_confusionMatrix=True, print_sum_confusionMatrix=True, print_target_distribution=True, save_feature_impact_across_folds=True, th_val=0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kappa_scores = []
    accuracy_scores = []
    balanced_acc_scores = []
    precision_scores = []
    recall_scores = []
    all_preds = []
    all_probs = []
    all_y_true = []
    all_feature_importances = []
    fold_targets = []
    confusion_matrices = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_cv = X.iloc[train_index]
        X_val_cv = X.iloc[val_index]
        y_train_cv = y.iloc[train_index]
        y_val_cv = y.iloc[val_index]

        # Initialize lists for predictions and probabilities
        fold_preds = []
        fold_probs = []

        # Train and predict with each model
        for model in models:
            model.fit(X_train_cv, y_train_cv)
            
            if save_feature_impact_across_folds and hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                all_feature_importances.append(feature_importances)
            
            y_pred = model.predict(X_val_cv)
            y_prob = model.predict_proba(X_val_cv)[:, 1]  # Assuming binary classification

            fold_preds.append(y_pred)
            fold_probs.append(y_prob)

        # Stack the predictions into a matrix
        predictions = np.vstack(fold_preds).T

        # Majority voting for predictions
        ensemble_pred, _ = mode(predictions, axis=1)
        ensemble_pred = ensemble_pred.ravel()

        # Average probabilities for ensemble
        ensemble_prob = np.mean(fold_probs, axis=0)

        # Convert probabilities to binary predictions based on threshold
        y_pred_thresh = (ensemble_prob >= th_val).astype(int)
        
        # Evaluate the ensemble model
        kappa = cohen_kappa_score(y_val_cv, y_pred_thresh)
        kappa_scores.append(kappa)

        accuracy = accuracy_score(y_val_cv, y_pred_thresh)
        accuracy_scores.append(accuracy)

        balanced_acc = balanced_accuracy_score(y_val_cv, y_pred_thresh)
        balanced_acc_scores.append(balanced_acc)

        precision = precision_score(y_val_cv, y_pred_thresh)
        precision_scores.append(precision)
        
        recall = recall_score(y_val_cv, y_pred_thresh)
        recall_scores.append(recall)

        # Collect confusion matrix for each fold
        cm = confusion_matrix(y_val_cv, ensemble_pred)
        confusion_matrices.append(cm)

        # Collect predictions and targets for plotting later
        all_preds.append(ensemble_pred)
        all_probs.append(ensemble_prob)
        all_y_true.append(y_val_cv)
        fold_targets.append(pd.DataFrame({'Fold': fold, 'Target': y_val_cv}))

    # Aggregate results
    mean_kappa = np.mean(kappa_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_balanced_acc = np.mean(balanced_acc_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    fold_targets_df = pd.concat(fold_targets)

    # Aggregate confusion matrices by averaging and summing across folds
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    sum_confusion_matrix = np.sum(confusion_matrices, axis=0)

    if print_avg_confusionMatrix:
        show_ConfusionMatrix_folds(avg_confusion_matrix=avg_confusion_matrix)
    
    if print_sum_confusionMatrix:
        print("Sum of Confusion Matrices:")
        print(sum_confusion_matrix)
    
    if print_target_distribution:
        plot_target_distribution(fold_targets_df)

    return {
        'mean_kappa': mean_kappa,
        'mean_accuracy': mean_accuracy,
        'mean_balanced_acc': mean_balanced_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'all_preds': np.concatenate(all_preds),
        'all_probs': np.concatenate(all_probs),
        'all_y_true': np.concatenate(all_y_true),
        'feature_importances': np.array(all_feature_importances),
        'fold_targets': fold_targets_df,
        'confusion_matrices': confusion_matrices,
        'avg_confusion_matrix': avg_confusion_matrix,
        'sum_confusion_matrix': sum_confusion_matrix
    }


def cross_validate_model_using_StratifiedKFold(model, X, y, n_splits=5, random_state=None, print_avg_confusionMatrix=True, print_sum_confusionMatrix=True, print_target_distribution=True, save_feature_impact_across_folds=True, th_val=0.5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    kappa_scores = []
    accuracy_scores = []
    balanced_acc_scores = []
    precision_scores = []
    recall_scores = []
    all_preds = []
    all_probs = []
    all_y_true = []
    all_feature_importances = []
    fold_targets = []
    confusion_matrices = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train_cv = X.iloc[train_index]
        X_val_cv = X.iloc[val_index]
        y_train_cv = y.iloc[train_index]
        y_val_cv = y.iloc[val_index]

        # Train the model
        model.fit(X_train_cv, y_train_cv)

        # Save feature importances
        if save_feature_impact_across_folds:
            feature_importances = model.feature_importances_  # For models like RandomForest or XGBoost
            all_feature_importances.append(feature_importances)

        # Predict on validation set
        y_pred = model.predict(X_val_cv)
        y_prob = model.predict_proba(X_val_cv)[:, 1]  # Assuming binary classification

        # Convert probabilities to binary predictions based on threshold
        y_pred_thresh = (y_prob >= th_val).astype(int)

        # Evaluate model
        kappa = cohen_kappa_score(y_val_cv, y_pred)
        kappa_scores.append(kappa)

        accuracy = accuracy_score(y_val_cv, y_pred_thresh)
        accuracy_scores.append(accuracy)

        balanced_acc = balanced_accuracy_score(y_val_cv, y_pred_thresh)
        balanced_acc_scores.append(balanced_acc)

        precision = precision_score(y_val_cv, y_pred_thresh)
        precision_scores.append(precision)

        recall = recall_score(y_val_cv, y_pred_thresh)
        recall_scores.append(recall)

        # Collect confusion matrix for each fold
        cm = confusion_matrix(y_val_cv, y_pred)
        confusion_matrices.append(cm)

        # Collect predictions and targets for plotting later
        all_preds.append(y_pred)
        all_probs.append(y_prob)
        all_y_true.append(y_val_cv)
        fold_targets.append(pd.DataFrame({'Fold': fold, 'Target': y_val_cv}))

    # Aggregate results
    mean_kappa = np.mean(kappa_scores)
    mean_accuracy = np.mean(accuracy_scores)
    mean_balanced_acc = np.mean(balanced_acc_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    fold_targets_df = pd.concat(fold_targets)

    # Aggregate confusion matrices by averaging and summing across folds
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0)
    sum_confusion_matrix = np.sum(confusion_matrices, axis=0)

    if print_avg_confusionMatrix:
        show_ConfusionMatrix_folds(avg_confusion_matrix=avg_confusion_matrix)
    
    if print_sum_confusionMatrix:
        print("Sum of Confusion Matrices:")
        print(sum_confusion_matrix)
    
    if print_target_distribution:
        plot_target_distribution(fold_targets_df)

    return {
        'mean_kappa': mean_kappa,
        'mean_accuracy': mean_accuracy,
        'mean_balanced_acc': mean_balanced_acc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'all_preds': np.concatenate(all_preds),
        'all_probs': np.concatenate(all_probs),
        'all_y_true': np.concatenate(all_y_true),
        'feature_importances': np.array(all_feature_importances),
        'fold_targets': fold_targets_df,
        'confusion_matrices': confusion_matrices,
        'avg_confusion_matrix': avg_confusion_matrix,
        'sum_confusion_matrix': sum_confusion_matrix  # Sum of confusion matrices across folds
    }


def evaluate_model(y_test, y_pred, y_prob, threshold=0.7, print_metrics=False, print_report=False, show_confusion_mat=False):
    """Evaluate the model using various metrics and display results."""  
    # Apply threshold to predictions
    y_pred_threshold = y_prob > threshold
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_threshold)
    report = classification_report(y_test, y_pred_threshold)
    auc_roc = roc_auc_score(y_test, y_prob)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    
    # Compute confusion matrix
    # cm = confusion_matrix(y_test, y_pred_threshold)
    
    # Print results
    if print_metrics:
        print(f'Threshold: {threshold}')
        print(f'AUC-ROC: {round(auc_roc, 2)}')
        print(f'Accuracy: {round(accuracy, 2)}')
        print(f'Balanced Accuracy: {round(balanced_acc, 2)}')
        print(f"Precision (Test): {precision}")
        print(f"Recall (Test): {recall}")
        print('Classification Report (1-fold):')
        
    if print_report:
        print(report)

    if show_confusion_mat:
        print('Confusion Matrix (1-fold):')
        show_ConfusionMatrix_test(y_test, y_pred_threshold)
    
    # Return results including the confusion matrix
    return {
        'Threshold': threshold,
        'AUC-ROC': round(auc_roc, 2),
        'Accuracy': round(accuracy, 2),
        'Balanced Accuracy': round(balanced_acc, 2),
        'Precision': round(precision, 2),
        'Recall': round(recall, 2)
        }


def main():
    # Load and preprocess data
    df = load_data('data/startup_data.csv')
    
    # Add economic indicators
    df = add_nasdaq_annual_changes(df)
    indicator_code = 'NY.GDP.MKTP.KD.ZG'
    df = add_economic_indicators(df, indicator_code)
    indicator_code = 'SL.UEM.TOTL.ZS'
    df = add_economic_indicators(df, indicator_code)
    
    # Preprocess the data
    X, y = preprocess_data_classifier(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    
    # Train, predict, and evaluate models
    rfv2_clf = train_rfv2_model(X_train, y_train)
    rfv2_pred, rv2_prob = predict_model(rfv2_clf, X_test)
    evaluate_model(y_test, rfv2_pred, rv2_prob, threshold=0.7)
    
    xgb_clf = train_xgb_model(X_train, y_train)
    xgb_pred, xgb_prob = predict_model(xgb_clf, X_test)
    # evaluate_model(y_test, xgb_pred, xgb_prob, threshold=0.7)
    
    rf_clf = train_rf_model(X_train, y_train)
    rf_pred, rf_prob = predict_model(rf_clf, X_test)
    # evaluate_model(y_test, rf_pred, rf_prob)
    
    svm_clf = train_svm_model(X_train, y_train)
    svm_pred, svm_prob = predict_model(svm_clf, X_test)
    # evaluate_model(y_test, svm_pred, svm_prob)
    
    dt_clf = train_decision_tree_model(X_train, y_train)
    dt_pred, dt_prob = predict_model(dt_clf, X_test)
    evaluate_model(y_test, dt_pred, dt_prob, threshold=0.7)

if __name__ == "__main__":
    main()
