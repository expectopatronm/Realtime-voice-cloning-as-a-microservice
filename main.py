from fastapi import FastAPI

from api.routers import train_embedding
from config import settings

app = FastAPI(
    docs_url=f"{settings.BASE_URL}/docs",
    redoc_url=f"{settings.BASE_URL}/redoc",
    openapi_url=f"{settings.BASE_URL}/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.include_router(train_embedding.router)

@app.get(f"{settings.BASE_URL}/")
async def index():
    return {"message": "Hi, world."}
