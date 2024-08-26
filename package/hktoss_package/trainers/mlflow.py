import mlflow
import os
import os.path as path
from hktoss_package.models.base import BaseSKLearnModel
from sklearn.metrics import f1_score, roc_auc_score


class MLFlowTrainer:
    tracking_uri: str

    def __init__(self, tracking_uri: str = "Databricks", **kwargs) -> None:
        self.tracking_uri = tracking_uri
        if tracking_uri == "databricks":
            mlflow.login(backend="databricks")
        else:
            mlflow.set_tracking_uri(tracking_uri)

    def run_experiment(
        self, model: type[BaseSKLearnModel], X_train, y_train, X_test, y_test
    ):
        mlflow.set_experiment(f"{model.config['model_name']}")
        mlflow.autolog(
            log_datasets=True,
            log_model_signatures=True,
            log_models=True,
            disable=False,
        )
        with mlflow.start_run():
            # Train
            model.fit(X_train, y_train)

            # Evaluation
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "f1_score": f1_score(y_test, y_pred),
                "roc_auc_score": roc_auc_score(y_test, y_pred_proba),
            }
            mlflow.log_metrics(metrics)

            # Log model & artifacts to MLFlow
            model_path = f"{model.config['model_name']}.pkl"
            save_dir = ".cache"
            if not path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            model.export_pkl(save_dir=save_dir, model_name=model_path)
            mlflow.log_artifact(path.join(save_dir, model_path), artifact_path="models")
            mlflow.log_params(model.model.get_params())

            # Log the sklearn model and register as version 1
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="sklearn-model",
                input_example=X_train,
                registered_model_name=model.config["model_name"],
            )

        # Turn OFF logger until next run
        mlflow.autolog(disable=True)
        print("Experiment run completed and logged in MLFlow")
