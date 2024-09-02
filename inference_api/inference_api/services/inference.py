from typing import List

import mlflow.pyfunc as pyfunc
import mlflow.sklearn
import numpy as np
import sqlalchemy
from boto3 import client
from pandas import DataFrame, Series
from sqlalchemy import Engine, text

from inference_api.config import config
from inference_api.models import InferenceDto, InferenceResult


class InferenceService:
    db_uri: str | None
    engine: Engine
    # Best Model
    model_name: str
    # Cached Model
    cached_model_name: str

    def __init__(self):
        self.s3 = client(
            "s3",
            aws_access_key_id=config.aws_access_key_id,
            aws_secret_access_key=config.aws_secret_access_key,
            region_name=config.aws_default_region,
        )
        self.engine = sqlalchemy.create_engine(config.mlflow_db_uri)
        self.model_name = None
        self.model = None
        self.cached_model = None
        self.cached_model_name = None

    def _run_select_query(self, query: str, args: dict = None):
        """retrieve rows from DB."""
        with self.engine.connect() as conn:
            if args:
                result = conn.execute(text(query), args).fetchall()
            else:
                result = conn.execute(text(query)).fetchall()
            result = [row._mapping for row in result]
        return result

    def get_model_info(self, model_name: str):
        """Get the latest version's model info, given model name"""
        result = self._run_select_query(
            """select 
                mv.name, mv.version, mv.storage_location, mvt.value f1_score 
                from model_versions mv 
                left join model_version_tags mvt 
                    on mv.name = mvt.name 
                    and mv.version = mvt.version 
                where mvt.key = 'f1_score'
                    and mv.name = :m
                order by mv.version desc
                """,
            {"m": model_name},
        )
        return result[0]

    def get_all_model_info(self):
        """Get latest version info of multiple model names"""
        result = self._run_select_query(
            """with gr as (
                select name, max(version) latest_version from model_versions group by name
            )
            select mv.name, mv.version, mv.storage_location, mvt.value f1_score
            from gr
            inner join model_versions mv
                on gr.name = mv.name
                and mv.version = gr.latest_version
            left join model_version_tags mvt
                on mv.name = mvt.name
                and mv.version = mvt.version
            where mvt.key = 'f1_score'
            order by mv.name asc
            """,
        )
        return result

    def _download_model(self, uri: str):
        return mlflow.sklearn.load_model(model_uri=uri)

    def load_model_by_name(self, model_name: str):
        """load a model from S3 by name"""
        if self.cached_model_name != model_name:
            model_info = self.get_model_info(model_name)

            self.cached_model_name = model_name
            self.cached_model = self._download_model(model_info["storage_location"])

    def load_best_model(self):
        all_model_info = self.get_all_model_info()
        best_model = sorted(all_model_info, key=lambda x: x["f1_score"], reverse=True)[
            0
        ]

        self.model_name = best_model["name"]
        self.model = self._download_model(best_model["storage_location"])

    def _preprocess_input(self, input: DataFrame):
        features = [c for c in list(input.columns) if not c in ["NAME", "TARGET"]]
        X = input[features].copy()
        y = input["TARGET"]

        # enforce input schema
        numeric_cols = list(X.select_dtypes(include="number").columns)
        X[numeric_cols] = X[numeric_cols].astype(np.float64)

        return X, y

    def get_inference_results(
        self, inferenceDto: InferenceDto, model_name: str = None
    ) -> List[InferenceResult]:
        """run inference on either specified model or the best model"""

        # Preprocess
        input = DataFrame.from_records(inferenceDto.data)
        input = input.set_index("SK_ID_CURR")
        X, y = self._preprocess_input(input)

        # Setup model
        if model_name:
            self.load_model_by_name(model_name)
            pred_probs = self.cached_model.predict_proba(X)
            preds = np.argmax(pred_probs, axis=1)
        else:
            if not self.model:
                self.load_best_model()
            pred_probs = self.model.predict_proba(X)
            preds = np.argmax(pred_probs, axis=1)

        result: List[InferenceResult] = (
            DataFrame(
                {
                    "NAME": input["NAME"],
                    "pred_probs": Series(pred_probs[:, 1], index=y.index).astype(float),
                    "preds": Series(preds, index=y.index).astype(int),
                    "gt": y,
                }
            )
            .reset_index()
            .to_dict(orient="records")
        )

        return result


inferenceService = InferenceService()
