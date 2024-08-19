import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score
from eval import show_ConfusionMatrix_test, get_precision_and_recall
from addData import add_nasdaq_annual_changes, add_economic_indicators
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Preprocess the data: encoding categorical features, and scaling numerical features."""
    
    # Create label
    y = df['status'].map({'acquired': 1, 'closed': 0})
    
    # Drop unnecessary columns
    df = df.drop(columns=['status', 'founded_at', 'name', 'id', 'state_code', 'object_id', 'labels', 'closed_at', 'Unnamed: 0', 
                          'Unnamed: 6', 'zip_code', 'city', 'closed_at'])
    X = df
    
    # Identify categorical and numerical columns
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['number']).columns
    
    # OneHotEncode categorical columns
    encoder = OneHotEncoder(sparse=False)
    encoded_categorical = encoder.fit_transform(X[categorical_columns])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Standardize numerical columns
    scaler = StandardScaler()
    scaled_numerical = scaler.fit_transform(X[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_columns)
    
    # Combine encoded categorical and scaled numerical data
    processed_df = pd.concat([encoded_categorical_df, scaled_numerical_df], axis=1)
    
    # Optionally replace NaN values
    processed_df.fillna(processed_df.mean(), inplace=True)
    # Alternatively, you can use KNNImputer
    # knn_imputer = KNNImputer(n_neighbors=5)
    # processed_df = pd.DataFrame(knn_imputer.fit_transform(processed_df), columns=processed_df.columns)
    
    return processed_df, y


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

def evaluate_model(model, X_test, y_test, threshold=0.7):
    """Evaluate the model using various metrics and display results."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
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
    print(f'Threshold: {threshold}')
    print(f'AUC-ROC: {round(auc_roc, 2)}')
    print(f'Accuracy: {round(accuracy, 2)}')
    print(f'Balanced Accuracy: {round(balanced_acc, 2)}')
    print('Classification Report:')
    print(report)
    
    # Confusion Matrix
    show_ConfusionMatrix_test(y_test, y_pred_threshold)
    precision, recall = get_precision_and_recall(y_test, y_pred_threshold)
    print(f"Precision (Test): {precision}")
    print(f"Recall (Test): {recall}")


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
    
    # Train and evaluate models
    xgb_clf = train_xgb_model(X_train, y_train)
    evaluate_model(xgb_clf, X_test, y_test)
    
    rf_clf = train_rf_model(X_train, y_train)
    evaluate_model(rf_clf, X_test, y_test)
    
    svm_clf = train_svm_model(X_train, y_train)
    evaluate_model(svm_clf, X_test, y_test)

if __name__ == "__main__":
    main()
