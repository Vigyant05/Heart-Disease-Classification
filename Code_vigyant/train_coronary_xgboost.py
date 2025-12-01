import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score, average_precision_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import joblib

def load_and_preprocess(filepath, target_col):
    """
    Loads dataset, handles missing values, scales features, and encodes categorical features.
    """
    print(f"Processing {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None

    # Separate features and target
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in {filepath}")
        return None, None
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values
    # Numeric columns: impute with mean
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='ignore')

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        imputer_num = SimpleImputer(strategy='mean')
        X[numeric_cols] = imputer_num.fit_transform(X[numeric_cols])
        
        # Scale numeric features
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Categorical columns: impute with mode and then label encode
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        imputer_cat = SimpleImputer(strategy='most_frequent')
        X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])
        
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target if it's categorical
    if y.dtype == 'object' or len(np.unique(y)) > 50:
         le_y = LabelEncoder()
         y = le_y.fit_transform(y)

    return X, y

def train_evaluate_coronary():
    """
    Trains XGBoost model for Coronary Heart Disease, evaluates metrics, and saves the model.
    """
    filepath = "Dataset/coronary/coronary_clinical_scan_data.csv"
    target_col = "target"
    models_dir = "Code_vigyant/models"
    os.makedirs(models_dir, exist_ok=True)

    X, y = load_and_preprocess(filepath, target_col)
    if X is None or y is None:
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter tuning
    params = {
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'min_child_weight': [1, 3, 5, 7]
    }
    
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    print("Tuning hyperparameters for Coronary model (Optimizing for mAP)...")
    # Use RandomizedSearchCV with scoring='average_precision' to directly optimize mAP
    search = RandomizedSearchCV(
        xgb_clf, 
        param_distributions=params, 
        n_iter=50, # Increased iterations for better search
        scoring='average_precision', # Optimize for mAP
        cv=5, 
        verbose=1, 
        random_state=42, 
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    
    model = search.best_estimator_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV mAP: {search.best_score_:.4f}")

    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted')
    map_score = average_precision_score(y_test, y_proba)

    print(f"--- Results for Coronary Heart Disease ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"mAP: {map_score:.4f}")
    print("-" * 30)

    # Save model as PKL
    model_path = os.path.join(models_dir, "coronary_xgboost_dedicated.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}\n")

if __name__ == "__main__":
    train_evaluate_coronary()
