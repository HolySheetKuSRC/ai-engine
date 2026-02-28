import os
import time
import logging
import asyncio
from openai import AsyncOpenAI
from src.config import settings
from google import genai

logger = logging.getLogger(__name__)

TYPHOON_COOLDOWN_UNTIL = 0
COOLDOWN_MINUTES = 10

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL
)

# Initialize Gemini Client (automatically uses GEMINI_API_KEY from env if available)
gemini_client = genai.Client() if settings.GEMINI_API_KEY else None

async def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using Typhoon ASR API.
    Falls back to Gemini 2.5 Flash if Typhoon fails.
    
    Args:
        file_path (str): Path to the audio file.
        
    Returns:
        str: Transcribed text.
    """
    global TYPHOON_COOLDOWN_UNTIL
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    if time.time() < TYPHOON_COOLDOWN_UNTIL:
        logger.warning("Typhoon is in cooldown. Routing directly to Gemini.")
        try:
            return await _transcribe_audio_gemini(file_path)
        except Exception as gemini_e:
            logger.error(f"Gemini Fallback ASR failed during cooldown: {gemini_e}")
            raise Exception(f"Fallback (Gemini) ASR failed while Typhoon was in cooldown. (Gemini error: {gemini_e})")

    try:
        with open(file_path, "rb") as audio_file:
            transcription = await client.audio.transcriptions.create(
                model="typhoon-asr-realtime",
                file=("file.mp3", audio_file, "audio/mp3") 
            )
        return transcription.text
    except Exception as e:
        TYPHOON_COOLDOWN_UNTIL = time.time() + (COOLDOWN_MINUTES * 60)
        print(f"Typhoon ASR failed. Tripping circuit breaker for {COOLDOWN_MINUTES} minutes.")
        logger.warning(f"Typhoon ASR failed: {e}. Attempting fallback to Gemini 2.5 Flash.")
        try:
            return await _transcribe_audio_gemini(file_path)
        except Exception as gemini_e:
            logger.error(f"Gemini Fallback ASR failed: {gemini_e}")
            raise Exception(f"Both primary (Typhoon) and fallback (Gemini) ASR failed. (Typhoon error: {e}, Gemini error: {gemini_e})")

async def _transcribe_audio_gemini(file_path: str) -> str:
    """
    Fallback transcription using Google Gemini 2.5 Flash API.
    Includes exponential backoff for transient 5xx errors.
    """
    if not gemini_client:
        raise ValueError("Gemini API Key is not configured.")

    max_retries = 3
    gemini_success = False
    transcribed_text = ""
    
    for attempt in range(max_retries):
        audio_file = None
        try:
            # Upload the audio file to Google (offload blocking IO)
            audio_file = await asyncio.to_thread(
                gemini_client.files.upload, 
                file=file_path
            )
            
            # Generate transcription with a strict prompt
            prompt = "Transcribe the following Thai audio exactly as spoken. Output ONLY the raw transcribed text. Do not use any markdown formatting, and do not add any conversational introductions or conclusions."
            
            # Use threaded call for synchronous generate_content
            response = await asyncio.to_thread(
                gemini_client.models.generate_content,
                model='gemini-2.5-flash',
                contents=[prompt, audio_file]
            )
            
            transcribed_text = response.text if response.text else ""
            transcribed_text = transcribed_text.strip().removeprefix('```text').removesuffix('```').strip()
            gemini_success = True
            break  # Break out of the retry loop if successful
            
        except Exception as e:
            error_str = str(e)
            if "503" in error_str or "500" in error_str or "502" in error_str or "504" in error_str:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Wait 1s, then 2s
                    logger.warning(f"Gemini API transient error ({error_str}). Retrying in {wait_time} seconds (Attempt {attempt + 1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                    continue # Try again
            
            # If it's a 4xx error (like 400 or 404) or we ran out of retries, raise the exception
            raise e
            
        finally:
            # IMPORTANT: Clean up the file from Google's servers
            if audio_file:
                try:
                    await asyncio.to_thread(gemini_client.files.delete, name=audio_file.name)
                except Exception as cleanup_e:
                    logger.error(f"Failed to cleanup Gemini file {audio_file.name}: {cleanup_e}")

    if not gemini_success:
        raise Exception("Gemini fallback failed after maximum retries.")

    return transcribed_text
