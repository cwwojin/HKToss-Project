import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


class CatBoostModel:
    """
    CatBoost Model
    ---
    Parameters:
        iterations : Number of boosting iterations (default: 1000)
        learning_rate : Step size shrinkage used in update to prevents overfitting (default: 0.1)
        depth : Depth of the tree (default: 6)
        class_weights : Weighting of classes (default: None)

    Methods:
        fit(X, y) : Fits the CatBoost model on the training data
        predict(X) : Predicts class labels for samples in X
        predict_proba(X) : Predicts class probabilities for samples in X
        evaluate(X, y) : Evaluates the model on the test data and prints metrics
    """

    def __init__(self, iterations=1000, learning_rate=0.1, depth=6, class_weights=None):
        self.scaler = StandardScaler()
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            class_weights=class_weights,
            verbose=0,
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)[:, 1]
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_pred_proba)
        cm = confusion_matrix(y, y_pred)

        print(f"CatBoost AUC: {auc:.4f}")
        print(f"CatBoost Accuracy: {accuracy:.4f}")
        print(f"CatBoost Precision: {precision:.4f}")
        print(f"CatBoost Recall: {recall:.4f}")
        print(f"CatBoost F1 Score: {f1:.4f}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
