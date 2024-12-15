import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np

def load_and_aggregate_data(good_dir, less_good_dir):
    """
    Load and aggregate data from good and less good directories.
    Filter data for 'lat' jump_type, drop unnecessary columns, and compute aggregate features.
    Args:
        good_dir (str): Path to the directory containing 'good' CSV files.
        less_good_dir (str): Path to the directory containing 'less good' CSV files.
    Returns:
        pd.DataFrame: Aggregated DataFrame with features and labels.
    """
    aggregated_data = []  # List to store aggregated data

    def aggregate_file(df, label):
        """
        Aggregate features for a single file into a single row.
        Args:
            df (pd.DataFrame): The input DataFrame for one file.
            label (int): The label for the data (1 for good, 0 for less good).
        Returns:
            pd.DataFrame: Single-row DataFrame with aggregated features and label.
        """
        # Drop unnecessary columns
        df = df.drop(columns=['jump_type', 'trial_number', 'Frame'], errors='ignore')

        # Aggregate features
        aggregated_stats = {}
        for column in df.columns:
            aggregated_stats[f"{column}_mean"] = df[column].mean()
            aggregated_stats[f"{column}_max"] = df[column].max()
            aggregated_stats[f"{column}_min"] = df[column].min()
            aggregated_stats[f"{column}_std"] = df[column].std()

        # Add the label
        aggregated_stats['Label'] = label

        # Return as a DataFrame (single row)
        return pd.DataFrame([aggregated_stats])

    # Process good files
    for file in os.listdir(good_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(good_dir, file)
            df = pd.read_csv(file_path)

            # Filter for 'lat' in jump_type
            df = df[df['jump_type'] == 'lat']
            if not df.empty:
                aggregated = aggregate_file(df, label=1)  # Label 1 for good
                aggregated_data.append(aggregated)

    # Process less good files
    for file in os.listdir(less_good_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(less_good_dir, file)
            df = pd.read_csv(file_path)

            # Filter for 'lat' in jump_type
            df = df[df['jump_type'] == 'lat']
            if not df.empty:
                aggregated = aggregate_file(df, label=0)  # Label 0 for less good
                aggregated_data.append(aggregated)

    # Combine all aggregated data
    final_data = pd.concat(aggregated_data, ignore_index=True)

    # Print class distributions
    print("Class Distribution in Aggregated Data:")
    print(final_data['Label'].value_counts())
    return final_data

def train_validate_test_split(X, y):
    """
    Split data into train (80%), validation (10%), and test (10%) sets.
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Labels.
    Returns:
        tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # First split into train and temp (90% train + 10% test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Then split temp into validation (10%) and test (10%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_with_cross_validation(X_train, y_train):
    """
    Train and evaluate models using 10-fold cross-validation.
    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.Series): Training labels.
    """
    # Define 10-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kfold, scoring='accuracy')
    print("Random Forest 10-Fold Cross-Validation Results:")
    print("Accuracy per Fold:", rf_cv_scores)
    print("Mean Accuracy: {:.2f}%".format(rf_cv_scores.mean() * 100))
    print("Standard Deviation: {:.2f}%".format(rf_cv_scores.std() * 100))

    # XGBoost
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kfold, scoring='accuracy')
    print("XGBoost 10-Fold Cross-Validation Results:")
    print("Accuracy per Fold:", xgb_cv_scores)
    print("Mean Accuracy: {:.2f}%".format(xgb_cv_scores.mean() * 100))
    print("Standard Deviation: {:.2f}%".format(xgb_cv_scores.std() * 100))

def train_and_evaluate_final(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Train models on train data and evaluate on validation and test sets.
    Args:
        X_train, X_val, X_test (pd.DataFrame): Feature matrices.
        y_train, y_val, y_test (pd.Series): Labels.
    """
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)

    print("\nRandom Forest Results")
    print("Validation Accuracy:", accuracy_score(y_val, rf_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, rf_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, rf_test_pred))

    # XGBoost
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_val_pred = xgb_model.predict(X_val)
    xgb_test_pred = xgb_model.predict(X_test)

    print("\nXGBoost Results")
    print("Validation Accuracy:", accuracy_score(y_val, xgb_val_pred))
    print("Test Accuracy:", accuracy_score(y_test, xgb_test_pred))
    print("Test Classification Report:\n", classification_report(y_test, xgb_test_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models on good and less good datasets.")
    parser.add_argument("good_dir", type=str, help="Path to the directory containing 'good' CSV files.")
    parser.add_argument("less_good_dir", type=str, help="Path to the directory containing 'less good' CSV files.")
    args = parser.parse_args()
    good_dir = args.good_dir
    less_good_dir = args.less_good_dir

    # Load and aggregate data
    aggregated_data = load_and_aggregate_data(good_dir, less_good_dir)

    # Separate features and labels
    X = aggregated_data.drop(columns=['Label'])
    y = aggregated_data['Label']

    # Normalize features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = train_validate_test_split(X, y)

    # Perform 10-fold cross-validation on training data
    train_with_cross_validation(X_train, y_train)

    # Train on train set and evaluate on validation and test sets
    train_and_evaluate_final(X_train, X_val, X_test, y_train, y_val, y_test)
