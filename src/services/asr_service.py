import os
from openai import AsyncOpenAI

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL
)

async def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using Typhoon ASR API.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    with open(file_path, "rb") as audio_file:
        transcription = await client.audio.transcriptions.create(
            model="typhoon-asr-realtime",
            file=audio_file
        )
    
    return transcription.text
