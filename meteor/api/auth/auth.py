import os

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer

from meteor import constants

API_SECRET_TOKEN = os.environ.get("API_SECRET_TOKEN", constants.DEFAULT_API_SECRET_TOKEN)

oauth2_scheme = HTTPBearer()  # use token authentication


def api_key_auth(api_key=Depends(oauth2_scheme)):
    api_keys = [API_SECRET_TOKEN]
    if api_key.credentials not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization Failed"
        )