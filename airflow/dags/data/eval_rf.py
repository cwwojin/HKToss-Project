import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


def evaluate_randomforest():
    # Load the trained model from the file
    model_filename = ".cache/trained_random_forest_model.pkl"
    with open(model_filename, "rb") as model_file:
        best_clf = pickle.load(model_file)

    # Load the cached data again to prepare the test set
    cache_filename = ".cache/train_data_cache.pkl"
    with open(cache_filename, "rb") as cache_file:
        df = pickle.load(cache_file)

    # Prepare the data
    df = df.drop(columns=["_id"])
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    # Split the data into training and testing sets
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Make predictions with the loaded model
    y_pred = best_clf.predict(X_test)
    y_proba = best_clf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print(f"ROC AUC Score: {roc_auc:.2f}")
    print("Classification Report:")
    print(report)
