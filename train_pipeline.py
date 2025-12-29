"""
Complete training pipeline for CI/CD
Trains both Logistic Regression and Random Forest models
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add heart_disease_code to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'heart_disease_code'))

def main():
    print("=" * 70)
    print("STARTING ML TRAINING PIPELINE")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/7] Loading dataset...")
    columns = [
        "age", "sex", "cp", "trestbps", "chol",
        "fbs", "restecg", "thalach", "exang",
        "oldpeak", "slope", "ca", "thal", "target"
    ]
    df = pd.read_csv('./heart_disease_dataset/processed.cleveland.data', header=None, names=columns)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Clean missing values
    print("\n[2/7] Cleaning missing values...")
    df.replace("?", np.nan, inplace=True)
    df = df.astype(float)
    missing_before = df.isnull().sum().sum()
    
    # 3. Impute missing values
    print("\n[3/7] Imputing missing values...")
    imputer = SimpleImputer(strategy="median")
    df[["ca", "thal"]] = imputer.fit_transform(df[["ca", "thal"]])
    print(f"Missing values imputed: {missing_before} -> {df.isnull().sum().sum()}")
    
    # Save imputer
    with open('imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)
    print("Imputer saved: imputer.pkl")
    
    # 4. Convert target to binary
    print("\n[4/7] Converting target to binary classification...")
    df["target"] = (df["target"] > 0).astype(int)
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")
    
    # 5. Prepare features for both models
    print("\n[5/7] Preparing features...")
    
    # Version 1: One-hot encoded for Logistic Regression
    df_encoded = pd.get_dummies(df, columns=['ca', 'thal'], prefix=['ca', 'thal'], drop_first=True, dtype=float)
    X_encoded = df_encoded.drop('target', axis=1)
    y_encoded = df_encoded['target']
    
    # Version 2: Original for Random Forest
    X_original = df.drop('target', axis=1)
    y_original = df['target']
    
    print(f"Features prepared - Encoded: {X_encoded.shape}, Original: {X_original.shape}")
    
    # 6. Train models
    print("\n[6/7] Training models...")
    
    # Logistic Regression
    print("\n  Training Logistic Regression...")
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
        X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    X_train_lr_scaled = X_train_lr.copy()
    X_test_lr_scaled = X_test_lr.copy()
    X_train_lr_scaled[numerical_features] = scaler.fit_transform(X_train_lr[numerical_features])
    X_test_lr_scaled[numerical_features] = scaler.transform(X_test_lr[numerical_features])
    
    lr_model = LogisticRegression(random_state=42, max_iter=2000)
    lr_model.fit(X_train_lr_scaled, y_train_lr)
    
    # Evaluate LR
    y_pred_lr = lr_model.predict(X_test_lr_scaled)
    y_pred_proba_lr = lr_model.predict_proba(X_test_lr_scaled)[:, 1]
    cv_scores_lr = cross_val_score(lr_model, X_train_lr_scaled, y_train_lr, cv=5, scoring='accuracy')
    
    lr_metrics = {
        'accuracy': accuracy_score(y_test_lr, y_pred_lr),
        'precision': precision_score(y_test_lr, y_pred_lr),
        'recall': recall_score(y_test_lr, y_pred_lr),
        'f1_score': f1_score(y_test_lr, y_pred_lr),
        'roc_auc': roc_auc_score(y_test_lr, y_pred_proba_lr),
        'cv_mean': cv_scores_lr.mean(),
        'cv_std': cv_scores_lr.std()
    }
    
    print(f"    Accuracy: {lr_metrics['accuracy']:.4f}")
    print(f"    ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"    CV Mean: {lr_metrics['cv_mean']:.4f}")
    
    # Save LR model and scaler
    with open('logistic_regression_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)
    with open('scaler_linear.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("    Model saved: logistic_regression_model.pkl")
    print("    Scaler saved: scaler_linear.pkl")
    
    # Random Forest
    print("\n  Training Random Forest...")
    X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
        X_original, y_original, test_size=0.2, random_state=42, stratify=y_original
    )
    
    rf_model = RandomForestClassifier(random_state=42, n_estimators=200)
    rf_model.fit(X_train_rf, y_train_rf)
    
    # Evaluate RF
    y_pred_rf = rf_model.predict(X_test_rf)
    y_pred_proba_rf = rf_model.predict_proba(X_test_rf)[:, 1]
    cv_scores_rf = cross_val_score(rf_model, X_train_rf, y_train_rf, cv=5, scoring='accuracy')
    
    rf_metrics = {
        'accuracy': accuracy_score(y_test_rf, y_pred_rf),
        'precision': precision_score(y_test_rf, y_pred_rf),
        'recall': recall_score(y_test_rf, y_pred_rf),
        'f1_score': f1_score(y_test_rf, y_pred_rf),
        'roc_auc': roc_auc_score(y_test_rf, y_pred_proba_rf),
        'cv_mean': cv_scores_rf.mean(),
        'cv_std': cv_scores_rf.std()
    }
    
    print(f"    Accuracy: {rf_metrics['accuracy']:.4f}")
    print(f"    ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print(f"    CV Mean: {rf_metrics['cv_mean']:.4f}")
    
    # Save RF model
    with open('random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("    Model saved: random_forest_model.pkl")
    
    # 7. Save configuration
    print("\n[7/7] Saving configuration...")
    config = {
        'missing_value_strategy': 'median imputation for ca and thal',
        'target_encoding': 'binary (0: no disease, 1: disease)',
        'categorical_encoding': 'one-hot encoding for ca and thal (drop_first=True)',
        'numerical_features': ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'],
        'categorical_features': ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'],
        'scaling': 'StandardScaler for numerical features (Logistic Regression only)',
        'test_size': 0.2,
        'random_state': 42,
        'logistic_regression_params': {'max_iter': 2000, 'random_state': 42},
        'random_forest_params': {'n_estimators': 200, 'random_state': 42},
        'logistic_regression_metrics': lr_metrics,
        'random_forest_metrics': rf_metrics
    }
    
    with open('preprocessing_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Configuration saved: preprocessing_config.json")
    
    # Summary
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("\nGenerated Files:")
    print("  - logistic_regression_model.pkl")
    print("  - random_forest_model.pkl")
    print("  - scaler_linear.pkl")
    print("  - imputer.pkl")
    print("  - preprocessing_config.json")
    
    print("\nModel Performance Summary:")
    print(f"  Logistic Regression - Accuracy: {lr_metrics['accuracy']:.4f}, ROC-AUC: {lr_metrics['roc_auc']:.4f}")
    print(f"  Random Forest       - Accuracy: {rf_metrics['accuracy']:.4f}, ROC-AUC: {rf_metrics['roc_auc']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
