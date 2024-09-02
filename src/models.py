from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # Import DecisionTreeClassifier
from eval import show_ConfusionMatrix_test, get_precision_and_recall
from getdata import add_nasdaq_annual_changes, add_economic_indicators
from preprocess import preprocess_data, load_data

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

def evaluate_model(y_test, y_pred, y_prob, threshold=0.7):
    """Evaluate the model using various metrics and display results."""  
    # Apply threshold to predictions
    y_pred_threshold = y_pred > threshold
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred_threshold)
    report = classification_report(y_test, y_pred_threshold)
    auc_roc = roc_auc_score(y_test, y_prob)
    balanced_acc = balanced_accuracy_score(y_test, y_pred_threshold)
    precision = precision_score(y_test, y_pred_threshold)
    recall = recall_score(y_test, y_pred_threshold)
    
    # Print results
    # print(f'Threshold: {threshold}')
    # print(f'AUC-ROC: {round(auc_roc, 2)}')
    # print(f'Accuracy: {round(accuracy, 2)}')
    # print(f'Balanced Accuracy: {round(balanced_acc, 2)}')
    # print('Classification Report:')
    # print(report)
    
    # Confusion Matrix
    show_ConfusionMatrix_test(y_test, y_pred_threshold)
    # precision, recall = get_precision_and_recall(y_test, y_pred_threshold)
    print(f"Precision (Test): {precision}")
    print(f"Recall (Test): {recall}")


    # Return results
    return {
        'Accuracy': accuracy,
        'AUC-ROC': auc_roc,
        'Balanced Accuracy': balanced_acc,
        'Precision': precision,
        'Recall': recall
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
    X, y = preprocess_data(df)
    
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
