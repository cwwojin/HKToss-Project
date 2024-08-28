import pickle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def evaluate_linear_regression():
    # Load the trained model from the file
    model_filename = ".cache/trained_linear_regression_model.pkl"
    with open(model_filename, "rb") as model_file:
        best_reg = pickle.load(model_file)

    # Load the cached data again to prepare the test set
    cache_filename = ".cache/train_data_cache.pkl"
    with open(cache_filename, "rb") as cache_file:
        df = pickle.load(cache_file)

    # Prepare the data
    df = df.drop(columns=["_id"])  # Assuming _id column is to be dropped
    X = df.drop("TARGET", axis=1)
    y = df["TARGET"]

    # Split the data into training and testing sets
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Make predictions with the loaded model
    y_pred = best_reg.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}")