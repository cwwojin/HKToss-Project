import os
from fastapi.security.api_key import APIKeyHeader
from fastapi import Security, HTTPException
from starlette.status import HTTP_401_UNAUTHORIZED
from inference_api.config import config

api_key_header = APIKeyHeader(name="access_token", auto_error=False)


async def ApiKeyGuard(api_key_header: str = Security(api_key_header)):
    """Api Key Guard"""
    if api_key_header == config.api_key:
        return api_key_header
    else:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED, detail="API KEY validation failed."
        )
