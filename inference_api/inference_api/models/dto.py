from pydantic import BaseModel
from typing import List, Dict


class InferenceDto(BaseModel):
    data: List[Dict] | None = None
