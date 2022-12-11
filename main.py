from fastapi import FastAPI

from api.routers import (
    faq_dataset_generator,
    faq_inference,
    squad_style_dataset_generator,
    squad_style_inference,
)
from config import settings

app = FastAPI(
    docs_url=f"{settings.BASE_URL}/docs",
    redoc_url=f"{settings.BASE_URL}/redoc",
    openapi_url=f"{settings.BASE_URL}/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

# app.include_router(faq_dataset_generator.router)
# app.include_router(faq_inference.router)
# app.include_router(squad_style_dataset_generator.router)
# app.include_router(squad_style_inference.router)


@app.get(f"{settings.BASE_URL}/")
async def index():
    return {"message": "Hi, world."}
