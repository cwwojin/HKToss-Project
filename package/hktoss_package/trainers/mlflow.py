import json
import os
import os.path as path
from datetime import datetime

import mlflow
import numpy as np
from hktoss_package.models import (
    CatBoostPipeline,
    LGBMPipeline,
    LogisticRegressionPipeline,
    MLPPipeline,
    RandomForestPipeline,
    XGBPipeline,
)
from hktoss_package.utils.functions import is_bool_col
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from mlflow.client import MlflowClient
from pandas import DataFrame
from sklearn.compose import make_column_selector
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from yacs.config import CfgNode as CN

MODEL_IMPROVEMENT_THR = 1e-4


class MLFlowTrainer:
    tracking_uri: str
    config: CN
    model_name: str

    def __init__(self, tracking_uri: str, config: CN) -> None:
        self.model = None
        self.config = config
        self.tracking_uri = tracking_uri
        if tracking_uri == "databricks":
            mlflow.login(backend="databricks")
        else:
            mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient(mlflow.get_tracking_uri())

    def _get_param_grid(self):
        if self.config:
            param_grid = dict(self.config[self.config.MODEL_TYPE.upper()])
            return {f"classifier__{k}": v for k, v in param_grid.items()}

    def prepare_model(
        self,
    ):
        self.model_name = f"{self.config.MODEL_TYPE}"
        if self.config.MODEL_TYPE == "logistic":
            model = LogisticRegressionPipeline()
        elif self.config.MODEL_TYPE == "randomforest":
            model = RandomForestPipeline()
        elif self.config.MODEL_TYPE == "xgboost":
            model = XGBPipeline()
        elif self.config.MODEL_TYPE == "lightgbm":
            model = LGBMPipeline()
        elif self.config.MODEL_TYPE == "catboost":
            model = CatBoostPipeline()
        elif self.config.MODEL_TYPE == "mlp":
            model = MLPPipeline()
        else:
            raise NotImplementedError(f"unrecognized model : {self.config.MODEL_TYPE}")

        self.model = model

    def prepare_data(self, df: DataFrame, grid_search: bool = False):
        id_col = self.config.DATASET.ID_COL_NAME
        target_col = self.config.DATASET.TARGET_COL_NAME
        self.dataframe = df.set_index(id_col)

        # column selection, ordering
        df_y = self.dataframe[target_col]
        df_x = self.dataframe.drop(columns=[target_col])
        df_x = df_x[sorted(list(df_x.columns))]

        # Column types
        one_hot_columns = [c for c in df_x.columns if is_bool_col(df_x[c])]
        numeric_columns = [c for c in df_x.columns if not c in one_hot_columns]

        # Data Sampler
        sampler = self.config.DATASET.SAMPLER
        if sampler == "under_random":
            data_sampler = RandomUnderSampler(
                random_state=self.config.DATASET.RANDOM_STATE, replacement=False
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "under_tomeks":
            data_sampler = TomekLinks()
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "over_random":
            data_sampler = RandomOverSampler(
                random_state=self.config.DATASET.RANDOM_STATE
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "over_smote":
            data_sampler = SMOTE(
                random_state=self.config.DATASET.RANDOM_STATE, k_neighbors=5
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)

        # Don't use split if using grid-search
        if grid_search:
            return df_x, df_y, {"bool": one_hot_columns, "num": numeric_columns}
        else:

            # dataset split
            X_train, X_test, y_train, y_test = train_test_split(
                df_x,
                df_y,
                test_size=self.config.DATASET.TEST_SIZE,
                random_state=self.config.DATASET.RANDOM_STATE,
                stratify=df_y,
            )

            return (
                X_train,
                X_test,
                y_train,
                y_test,
                {"bool": one_hot_columns, "num": numeric_columns},
            )

    def run_experiment(self, dataframe: DataFrame):
        if self.config.GRID_SEARCH:
            self.run_grid_search_experiment(dataframe)
        else:
            self.run_experiment_single(dataframe)

    def run_experiment_single(self, dataframe: DataFrame):
        # prepare dataset
        X_train, X_test, y_train, y_test, column_types = self.prepare_data(df=dataframe)

        # load model
        if not self.model:
            self.prepare_model()

        # self.model.pipeline = self.model.build_pipe_transformer(
        #     column_types=column_types
        # )

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
            model_file = f"{self.model_name}.pkl"
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
                sk_model=self.model.pipeline,
                artifact_path="registered-model",
                registered_model_name=self.model_name,
            )

        # Turn OFF logger until next run
        mlflow.autolog(disable=True)
        print("Experiment run completed and logged in MLFlow")

    def run_grid_search_experiment(self, dataframe: DataFrame):
        # prepare dataset
        X, y, column_types = self.prepare_data(df=dataframe, grid_search=True)

        # load model
        if not self.model:
            self.prepare_model()

        # self.model.pipeline = self.model.build_pipe_transformer(
        #     column_types=column_types
        # )

        # init experiment
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S")
        experiment_name = f"{self.config.LOGGER.EXPERIMENT_NAME if self.config.LOGGER.EXPERIMENT_NAME else self.config.MODEL_TYPE}_gridsearch"
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
        except:
            experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        mlflow.set_experiment(experiment_id=experiment_id)
        mlflow.autolog(
            log_model_signatures=True,
            log_models=False,
            log_datasets=False,
            disable=False,
        )
        run_name = f"{self.config.LOGGER.RUN_NAME if self.config.LOGGER.RUN_NAME else 'run'}_{timestamp}"
        with mlflow.start_run(run_name=run_name):

            # Init grid search
            param_grid = self._get_param_grid()
            if hasattr(self.model, "pca"):
                param_grid["pca__n_components"] = self.config.PCA.N_COMPONENTS + [None]
            search = GridSearchCV(
                self.model.pipeline,  # Fit to model.pipeline, not model
                param_grid,
                cv=int(1 / self.config.DATASET.TEST_SIZE),
                scoring={
                    "f1_score": "f1_macro",
                    "f1_score_micro": "f1_micro",
                    "recall": "recall",
                    "roc_auc_score": "roc_auc",
                },
                refit="f1_score",
                n_jobs=os.cpu_count() if self.config.MULTIPROCESSING else None,
            )

            # Fit
            search.fit(X, y)

            # Save only the best estimator
            self.model.pipeline = search.best_estimator_

            # Log model & experiment info
            mlflow.log_params(dict(self.config))
            mlflow.log_params(search.best_params_)

            # Log evaluation metrics
            mlflow.log_metrics({"f1_score_best": search.best_score_})
            save_dir = ".cache"
            if not path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            csv_path = path.join(save_dir, "cv_results.csv")
            DataFrame(search.cv_results_).to_csv(csv_path)
            mlflow.log_artifact(csv_path, artifact_path="grid_search")

            # Log model & artifacts to MLFlow
            model_file = f"{self.model_name}.pkl"
            model_path = path.join(save_dir, model_file)

            self.model.export_pkl(model_path)
            mlflow.log_artifact(model_path, artifact_path="model_pkl")

            # Delete cached files
            os.remove(model_path)
            os.remove(csv_path)

            # Register the model only if score is improved
            prev_version = self.client.get_latest_versions(self.model_name)[0]
            prev_best = (
                prev_version.tags["f1_score"] if "f1_score" in prev_version.tags else 0
            )

            if (
                np.float32(search.best_score_) - np.float32(prev_best)
            ) > MODEL_IMPROVEMENT_THR:
                print(
                    "Trained model is better than the latest version. Saving model to the registry.."
                )

                # Log the sklearn model and register
                mlflow.sklearn.log_model(
                    sk_model=self.model.pipeline,
                    artifact_path="registered-model",
                    registered_model_name=self.model_name,
                    input_example=X.iloc[:1, :],
                )
                # Add Tags to registered model
                model_info = self.client.get_latest_versions(self.model_name)[0]
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_info.version,
                    key="f1_score",
                    value=search.best_score_,
                )
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_info.version,
                    key="params",
                    value=json.dumps(search.best_params_),
                )
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_info.version,
                    key="config",
                    value=json.dumps(dict(self.config)),
                )
            else:
                # Log the sklearn model, but NOT register
                mlflow.sklearn.log_model(
                    sk_model=self.model.pipeline,
                    artifact_path="registered-model",
                    input_example=X.iloc[:1, :],
                )

        # Turn OFF logger until next run
        mlflow.autolog(disable=True)
        print("Experiment run completed and logged in MLFlow")
