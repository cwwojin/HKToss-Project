from typing import List

from fastapi import Depends, FastAPI
from fastapi.security.api_key import APIKey
from starlette.status import HTTP_200_OK

from inference_api.config import config
from inference_api.models import InferenceDto, InferenceResult
from inference_api.services import inferenceService
from inference_api.utils import ApiKeyGuard

app = FastAPI()


@app.get("/health")
def health_check():
    """Health check"""
    return HTTP_200_OK


@app.post("/inference")
def get_inference_result(
    inferenceDto: InferenceDto,
    model_name: str = None,
    api_key: APIKey = Depends(ApiKeyGuard),
) -> List[InferenceResult]:
    """Run model inference."""
    return inferenceService.get_inference_results(inferenceDto, model_name)
