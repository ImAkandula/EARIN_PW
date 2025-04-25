import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from datasets import load_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# Create a directory for plots if it doesn't exist
os.makedirs("plots", exist_ok=True)


if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Load and preprocess Titanic dataset
    X, y = load_dataset("titanic")

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Define models
    model1 = LogisticRegression(max_iter=1000)
    model2 = RandomForestClassifier(random_state=seed)

    # Evaluate models using 4-fold cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring="accuracy")
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring="accuracy")

    print("Logistic Regression CV Accuracy:", scores1.mean())
    print("Random Forest CV Accuracy:", scores2.mean())

    # Fit the models on the training set
    final_model1 = model1.fit(X_train, y_train)
    final_model2 = model2.fit(X_train, y_train)

    # Predict on the test set
    predictions1 = final_model1.predict(X_test)
    predictions2 = final_model2.predict(X_test)

    # Evaluate predictions
    print("Logistic Regression Test Accuracy:", accuracy_score(y_test, predictions1))
    print("Random Forest Test Accuracy:", accuracy_score(y_test, predictions2))

    # --- Logistic Regression Detailed Report ---
    print("\n--- Logistic Regression Classification Report ---")
    print(classification_report(y_test, predictions1))

    print("\nConfusion Matrix (Logistic Regression):")
    conf_matrix1 = confusion_matrix(y_test, predictions1)
    sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.title("Logistic Regression Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/logistic_regression_confusion_matrix.png")
    plt.show()

    # --- Random Forest Detailed Report ---
    print("\n--- Random Forest Classification Report ---")
    print(classification_report(y_test, predictions2))

    print("\nConfusion Matrix (Random Forest):")
    conf_matrix2 = confusion_matrix(y_test, predictions2)
    sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Greens',
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    plt.title("Random Forest Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("plots/random_forest_confusion_matrix.png")
    plt.show()

    # --- Feature Importance (Random Forest only) ---
    importances = final_model2.feature_importances_
    features = X.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title("Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("plots/random_forest_feature_importance.png")
    plt.show()
