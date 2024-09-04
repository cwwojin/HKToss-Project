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
from imblearn.over_sampling import SMOTENC, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from mlflow.client import MlflowClient
from pandas import DataFrame
from sklearn.metrics import classification_report, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    TunedThresholdClassifierCV,
    train_test_split,
)
from yacs.config import CfgNode as CN

MODEL_IMPROVEMENT_THR = 1e-4
SAMPLER_TARGETS = {
    0: 150000,
    1: 100000,
}
CUSTOM_PR_WEIGHTS = {
    "precision": 1,
    "recall": 3,
}


def positive_f1(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=[0, 1], output_dict=True)[
        1
    ]["f1-score"]


def positive_recall(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=[0, 1], output_dict=True)[
        1
    ]["recall"]


def weighted_f1_custom(y_true, y_pred):
    cls_report = classification_report(
        y_true, y_pred, target_names=[0, 1], output_dict=True
    )
    precision = cls_report[1]["precision"]
    recall = cls_report[1]["recall"]
    divisor = (precision * CUSTOM_PR_WEIGHTS["precision"]) + (
        recall * CUSTOM_PR_WEIGHTS["recall"]
    )
    score = (
        (
            2
            * (
                precision
                * CUSTOM_PR_WEIGHTS["precision"]
                * recall
                * CUSTOM_PR_WEIGHTS["recall"]
            )
            / divisor
        )
        if divisor > 0
        else 0
    )
    return score


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
        param_grid = dict(self.config[self.config.MODEL_TYPE.upper()])

        return {f"classifier__{k}": v for k, v in param_grid.items()}

    def prepare_model(self, column_types: dict = None):
        self.model_name = f"{self.config.MODEL_TYPE}"
        if self.config.MODEL_TYPE == "logistic":
            model = LogisticRegressionPipeline(column_types)
        elif self.config.MODEL_TYPE == "randomforest":
            model = RandomForestPipeline(column_types)
        elif self.config.MODEL_TYPE == "xgboost":
            model = XGBPipeline(column_types)
        elif self.config.MODEL_TYPE == "lightgbm":
            model = LGBMPipeline(column_types)
        elif self.config.MODEL_TYPE == "catboost":
            model = CatBoostPipeline(column_types)
        elif self.config.MODEL_TYPE == "mlp":
            model = MLPPipeline(column_types)
        else:
            raise NotImplementedError(f"unrecognized model : {self.config.MODEL_TYPE}")

        self.model = model

    def prepare_data(self, df: DataFrame, grid_search: bool = False):
        id_col = self.config.DATASET.ID_COL_NAME
        target_col = self.config.DATASET.TARGET_COL_NAME
        self.dataframe = df.set_index(id_col)

        # column selection, ordering
        df_y = self.dataframe[target_col]
        df_y = df_y.astype(int)
        df_x = self.dataframe.drop(columns=[target_col])
        df_x = df_x[sorted(list(df_x.columns))]

        # Column types
        cat_columns = list(df_x.select_dtypes(exclude=["number", "bool"]).columns)
        bool_columns = list(df_x.select_dtypes(include="bool").columns)
        numeric_columns = list(df_x.select_dtypes(include="number").columns)
        column_types = {
            "cat": cat_columns,
            "bool": bool_columns,
            "num": numeric_columns,
        }

        # Convert bool & numeric -> float64
        df_x[numeric_columns] = df_x[numeric_columns].astype(np.float64)
        df_x[bool_columns] = df_x[bool_columns].astype(np.float64)

        # Train / Test Split
        df_x, df_x_test, df_y, df_y_test = train_test_split(
            df_x,
            df_y,
            test_size=0.1,
            random_state=self.config.DATASET.RANDOM_STATE,
            stratify=df_y,
        )

        # Data Sampler
        sampler = self.config.DATASET.SAMPLER
        if sampler == "under_random":
            data_sampler = RandomUnderSampler(
                random_state=self.config.DATASET.RANDOM_STATE,
                replacement=False,
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "over_random":
            data_sampler = RandomOverSampler(
                random_state=self.config.DATASET.RANDOM_STATE
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "over_smote":
            data_sampler = SMOTENC(
                categorical_features=cat_columns,
                random_state=self.config.DATASET.RANDOM_STATE,
                k_neighbors=5,
            )
            df_x, df_y = data_sampler.fit_resample(df_x, df_y)
        elif sampler == "composite":
            # Oversampling
            over_sampler = SMOTENC(
                categorical_features=cat_columns,
                random_state=self.config.DATASET.RANDOM_STATE,
                sampling_strategy={
                    0: df_y.value_counts()[0],
                    1: SAMPLER_TARGETS[1],
                },
            )
            df_x, df_y = over_sampler.fit_resample(df_x, df_y)

            # Undersampling
            under_sampler = RandomUnderSampler(
                random_state=self.config.DATASET.RANDOM_STATE,
                replacement=False,
                sampling_strategy={
                    0: SAMPLER_TARGETS[0],
                    1: df_y.value_counts()[1],
                },
            )
            df_x, df_y = under_sampler.fit_resample(df_x, df_y)

        return df_x, df_x_test, df_y, df_y_test, column_types

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
        X, X_test, y, y_test, column_types = self.prepare_data(
            df=dataframe, grid_search=True
        )

        # load model
        if not self.model:
            self.prepare_model(column_types)

        # init experiment
        timestamp = datetime.strftime(datetime.now(), "%Y-%m-%d_%H:%M:%S")
        experiment_name = f"{'airflow_' if self.config.AIRFLOW else ''}{self.config.LOGGER.EXPERIMENT_NAME if self.config.LOGGER.EXPERIMENT_NAME else self.config.MODEL_TYPE}_gridsearch"
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
                if self.model.preprocessor:
                    param_grid["preprocessor__numeric__pca__n_components"] = (
                        self.config.PCA.N_COMPONENTS
                        if self.config.PCA.N_COMPONENTS
                        else [None]
                    )
                else:
                    param_grid["pca__n_components"] = self.config.PCA.N_COMPONENTS + [
                        None
                    ]
            search = GridSearchCV(
                self.model.pipeline,
                param_grid,
                cv=int(1 / self.config.DATASET.TEST_SIZE),
                scoring={
                    "f1_score_true": make_scorer(score_func=positive_f1),
                    "f1_score": "f1_macro",
                    "f1_score_micro": "f1_micro",
                    "recall": "recall",
                    "recall_true": make_scorer(score_func=positive_recall),
                    "roc_auc_score": "roc_auc",
                },
                refit="recall_true",
                verbose=1,
                n_jobs=os.cpu_count() if self.config.MULTIPROCESSING else None,
            )

            # 1. Fit - Grid Search CV
            search.fit(X, y)
            mlflow.autolog(disable=True)

            # Save only the best estimator
            self.model.pipeline = search.best_estimator_

            # 2. Fit - Tune Threshold CV
            if self.config.TUNE_THRESHOLD:
                tuned_model = TunedThresholdClassifierCV(
                    search.best_estimator_,
                    # scoring=make_scorer(score_func=positive_f1),
                    scoring="balanced_accuracy",
                    store_cv_results=True,
                    random_state=self.config.DATASET.RANDOM_STATE,
                    n_jobs=os.cpu_count() if self.config.MULTIPROCESSING else None,
                ).fit(X, y)

                print(
                    f"[TunedThresholdClassifierCV] Cut-off point found at {tuned_model.best_threshold_:.3f}"
                )

                self.model.pipeline = tuned_model

                # Log evaluation metrics - TunedThresholdClassifierCV
                mlflow.log_metrics({"tuned_score": tuned_model.best_score_})

                save_dir = ".cache"
                if not path.isdir(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                csv_path = path.join(save_dir, "cv_results_tuned.csv")
                DataFrame(tuned_model.cv_results_).to_csv(csv_path)
                mlflow.log_artifact(csv_path, artifact_path="tune_threshold")
                os.remove(csv_path)

                # Log best threshold
                mlflow.log_params({"tuned_threshold": tuned_model.best_threshold_})

                registry_dict = search.best_params_
                registry_dict["classifier__threshold"] = tuned_model.best_threshold_

            # Log model & experiment info
            mlflow.log_params(dict(self.config))

            # Log evaluation metrics
            mlflow.log_metrics({"gs_score_best": search.best_score_})

            save_dir = ".cache"
            if not path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            csv_path = path.join(save_dir, "cv_results.csv")
            DataFrame(search.cv_results_).to_csv(csv_path)
            mlflow.log_artifact(csv_path, artifact_path="grid_search")
            os.remove(csv_path)

            # Run Test Set
            test_cls_report = classification_report(
                y_test,
                self.model.pipeline.predict(X_test),
                output_dict=True,
            )
            print(test_cls_report)
            test_f1 = test_cls_report["macro avg"]["f1-score"]
            mlflow.log_metrics(
                {
                    "test_precision": test_cls_report["macro avg"]["precision"],
                    "test_recall": test_cls_report["macro avg"]["recall"],
                    "test_f1_score": test_f1,
                }
            )

            # Register the model as new version only if score is improved
            try:
                prev_version = self.client.get_latest_versions(self.model_name)[0]
                prev_best = (
                    prev_version.tags["f1_score"]
                    if "f1_score" in prev_version.tags
                    else 0
                )
            except:
                prev_best = 0

            if (np.float32(test_f1) - np.float32(prev_best)) > MODEL_IMPROVEMENT_THR:
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
                    value=test_f1,
                )
                self.client.set_model_version_tag(
                    name=self.model_name,
                    version=model_info.version,
                    key="params",
                    value=(
                        json.dumps(registry_dict)
                        if self.config.TUNE_THRESHOLD
                        else json.dumps(search.best_params_)
                    ),
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
        mlflow.end_run()
        print("Experiment run completed and logged in MLFlow")
