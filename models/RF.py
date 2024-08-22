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


class RandomForestModel:
    """
    Random Forest Model
    ---
    Parameters:
        n_estimators : Number of trees in the forest (default: 100)
        class_weight : Weighting of classes (default: 'balanced')

    Methods:
        fit(X, y) : Fits the random forest model on the training data
        predict(X) : Predicts class labels for samples in X
        predict_proba(X) : Predicts class probabilities for samples in X
        evaluate(X, y) : Evaluates the model on the test data and prints metrics
    """

    def __init__(self, n_estimators=100, class_weight="balanced"):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight
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

        print(f"Random Forest AUC: {auc:.4f}")
        print(f"Random Forest Accuracy: {accuracy:.4f}")
        print(f"Random Forest Precision: {precision:.4f}")
        print(f"Random Forest Recall: {recall:.4f}")
        print(f"Random Forest F1 Score: {f1:.4f}")

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()
