from pydantic import BaseModel

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from config import settings

from core.runner import squad_style_inference_runner

router = APIRouter(
    prefix=f"{settings.BASE_URL}",
    tags=["SQuAD Style Inference"],
)


class Input(BaseModel):
    text: str


@router.post("/squad-style-inference")
async def squad_style_inference(input: Input):
    """Infers an Input Question from the generated SQuaD Dataset.

    Args:
        Input: string.

    Returns:
        Output: html response.
    """

    input_txt = input.text

    context = r"""
    A new vehicle must be broken in within the first 1,000 miles (1,500 km) 
    so that all moving parts work smoothly together, which helps to increase the 
    service life of the engine and other drive components. 
    """

    response_dict = {}
    response_dict[input_txt], indices = squad_style_inference_runner(
        input_txt, context
    )

    highlight_start = "<mark><b>"
    highlight_end = "</b></mark>"

    html_text = (
        context[: indices[0]]
        + highlight_start
        + context[indices[0] : indices[1]]
        + context[indices[1] :]
        + highlight_end
    )

    html_content = """<html>
                        <h3>{}</h3>
                        <br />
                        <body>
                            {}
                        </body>
                    </html>""".format(
        input_txt, html_text
    )

    return HTMLResponse(content=html_content)
