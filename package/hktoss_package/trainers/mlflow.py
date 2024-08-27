import mlflow
import os
import os.path as path
from hktoss_package.models.base import BaseSKLearnPipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pandas import DataFrame
from yacs.config import CfgNode as CN
from hktoss_package.models import LogisticRegressionPipeline
from datetime import datetime


class MLFlowTrainer:
    tracking_uri: str
    config: CN
    model: type[BaseSKLearnPipeline]

    def __init__(self, tracking_uri: str, config: CN, **kwargs) -> None:
        self.model = None
        self.config = config
        self.tracking_uri = tracking_uri
        if tracking_uri == "databricks":
            mlflow.login(backend="databricks")
        else:
            mlflow.set_tracking_uri(tracking_uri)

    def prepare_model(self):
        model_name = f"{self.config.MODEL_TYPE}"
        if self.config.MODEL_TYPE == "logistic":
            model = LogisticRegressionPipeline(model_name)
        else:
            raise NotImplementedError(f"unrecognized model : {self.config.MODEL_TYPE}")

        self.model = model

    def prepare_data(self, df: DataFrame):
        id_col = self.config.DATASET.ID_COL_NAME
        target_col = self.config.DATASET.TARGET_COL_NAME
        self.dataframe = df.set_index(id_col)

        # column selection, ordering
        df_y = self.dataframe[target_col]
        df_x = self.dataframe.drop(columns=[target_col])
        df_x = df_x[sorted(list(df_x.columns))]

        # dataset split
        X_train, X_test, y_train, y_test = train_test_split(
            df_x,
            df_y,
            test_size=self.config.DATASET.TEST_SIZE,
            random_state=self.config.DATASET.RANDOM_STATE,
            stratify=df_y,
        )

        return X_train, X_test, y_train, y_test

    def run_experiment(self, dataframe: DataFrame):
        # load model
        if not self.model:
            self.prepare_model()

        # prepare dataset
        X_train, X_test, y_train, y_test = self.prepare_data(df=dataframe)

        # init experiment
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S")
        mlflow.set_experiment(
            experiment_name=(
                self.config.LOGGER.EXPERIMENT_NAME
                if self.config.LOGGER.EXPERIMENT_NAME
                else f"{self.config.MODEL_TYPE}"
            )
        )
        mlflow.autolog(
            log_model_signatures=True,
            log_models=False,
            log_datasets=False,
            disable=False,
        )
        run_name = f"{self.config.LOGGER.RUN_NAME if self.config.LOGGER.RUN_NAME else 'run'}_{timestamp}"
        with mlflow.start_run(run_name=run_name):
            # Train
            self.model.fit(X_train, y_train)

            # Evaluation
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            metrics = {
                "test_f1_score": f1_score(y_test, y_pred),
                "test_roc_auc_score": roc_auc_score(y_test, y_pred_proba),
            }
            mlflow.log_metrics(metrics)

            # Log model & artifacts to MLFlow
            model_file = f"{self.model.model_name}.pkl"
            save_dir = ".cache"
            model_path = path.join(save_dir, model_file)
            if not path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            self.model.export_pkl(model_path)
            mlflow.log_artifact(model_path, artifact_path="model_pkl")

            # Delete cached model
            os.remove(model_path)

            # Log the sklearn model and register
            mlflow.sklearn.log_model(
                sk_model=self.model,
                artifact_path="registered-model",
                registered_model_name=self.model.model_name,
            )

        # Turn OFF logger until next run
        mlflow.autolog(disable=True)
        print("Experiment run completed and logged in MLFlow")
