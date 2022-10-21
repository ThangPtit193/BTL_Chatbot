from http.client import HTTPException
import uvicorn
from fastapi import FastAPI, APIRouter
from fastapi.openapi.utils import get_openapi
from starlette.middleware.cors import CORSMiddleware

from core.config import settings
from rest_api.api_v1.api import api_router
from rest_api.error.http_error_handler import http_error_handler

try:
    from utils import __version__ as venus_version
except:
    venus_version = "0.0.0"  # only for development

router = APIRouter()


@router.get("/", status_code=200)
async def root():
    return {"message": "This is Venus Services"}


def get_application() -> FastAPI:
    application = FastAPI(
        title="Venus Services",
        debug=True,
        version=venus_version
    )

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.add_exception_handler(HTTPException, http_error_handler)
    application.include_router(api_router, prefix=settings.API_V1_STR)
    application.include_router(router)

    return application


def get_openapi_specs() -> dict:
    """
    Used to autogenerate OpenAPI specs file to use in the documentation.
    See `docs/_src/api/openapi/generate_openapi_specs.py`
    """
    app = get_application()
    return get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes
    )


app = get_application()

if __name__ == "__main__":
    # port = int(sys.argv[1])
    # Use this for debugging purposes only
    uvicorn.run("main:app", host="0.0.0.0", port=8891)
