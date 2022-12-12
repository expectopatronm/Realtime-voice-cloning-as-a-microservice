from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from config import settings

router = APIRouter(
    prefix=f"{settings.BASE_URL}",
    tags=["Train Embedding on Custom Voice"],
)


class Input(BaseModel):
    text: str


@router.post("/train-embedding-on-custom-voice")
async def train_embedding_on_custom_voice(input: Input):
    """Trains TTS Embedding on custom voice input.

    Args:
        Input: Audio File.

    Returns:
        Output: html response.
    """


    html_content = """<html>
                        <h3>Hello World</h3>
                        <br />
                        <body>
                        </body>
                    </html>"""

    return HTMLResponse(content=html_content)
