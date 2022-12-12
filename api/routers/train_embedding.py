from pydantic import BaseModel

import io
import soundfile as sf

import numpy as np

from fastapi import APIRouter, File
from fastapi.responses import Response

from config import settings

from core.runner import synthesize_speech

router = APIRouter(
    prefix=f"{settings.BASE_URL}",
    tags=["Train Embedding on Custom Voice"],
)


class Input(BaseModel):
    text: str


@router.post("/train-embedding-on-custom-voice")
async def train_embedding_on_custom_voice(audio: bytes = File(...)):
    """Trains TTS Embedding on custom voice input.
    
    Args:
        Input: Audio File.

    Returns:
        Output: Bytes.
    """

    audio_file, sample_rate = sf.read(io.BytesIO(audio))

    generated_wav, sample_rate = synthesize_speech(audio_file)

    # Output as memory file
    file_format = "WAV"
    memory_file = io.BytesIO( )
    memory_file.name = "generated_audio.wav"
    sf.write(memory_file, generated_wav.astype(np.float32), sample_rate, format=file_format)

    return Response(content=memory_file.getvalue())
