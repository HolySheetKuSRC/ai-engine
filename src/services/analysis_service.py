import os
import json
import asyncio
from openai import AsyncOpenAI
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typhoon Client (OpenAI-compatible)
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=BASE_URL,
    timeout=45.0,
    max_retries=0  # tenacity handles retries; SDK retries would double the wait
)

MODEL_NAME = "typhoon-v2.5-30b-a3b-instruct"

# System Prompt
SYSTEM_PROMPT = """
You are an AI Assistant for a Sheet Marketplace. 
Your task is to analyze the provided study material content (OCR text) and return a structured analysis in JSON format.
The JSON must contain:
1. "summary": A concise 2-3 sentence overview of the content (in Thai).
2. "assessment": A list of 3-5 short bullet points highlighting key topics or features (e.g., "เน้นโจทย์ Limit", "มีสูตรลัด").
3. "tags": A list of 5 relevant hashtags (e.g., ["#Calculus", "#Midterm", "#Note"]).

Return ONLY valid JSON. Do not include markdown formatting like ```json ... ```.
"""


async def analyze_sheet_content(full_text: str) -> dict:
    """
    Public entry point: Analyzes OCR text with chunking strategy if needed.
    """
    if not full_text:
        return _empty_result()
        
    # Increased chunk size to reduce the total number of API requests
    CHUNK_SIZE = 15000
    
    # If text is small, direct analysis
    if len(full_text) <= CHUNK_SIZE:
        return await _call_ai_api(full_text)
        
    # Chunking Strategy
    chunks = [full_text[i:i+CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
    logger.info(f"Text too large ({len(full_text)} chars). Splitting into {len(chunks)} chunks.")
    
    # Process chunks concurrently with rate limiting (max 3 concurrent)
    semaphore = asyncio.Semaphore(3)
    
    async def process_chunk_with_limit(chunk: str) -> dict:
        async with semaphore:
            # Add a polite delay to avoid burst limits
            await asyncio.sleep(2)
            return await _call_ai_api(chunk, is_partial=True)

    tasks = [process_chunk_with_limit(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Merge summaries
    combined_summary = "\n".join([r.get("summary", "") for r in results])
    
    # Final Pass
    logger.info("Running final analysis on merged summaries...")
    final_result = await _call_ai_api(combined_summary)
    
    # Merge tags/assessments if needed (optional, but let's trust the final pass to re-extract key points from the summary)
    # Alternatively, we could aggregate all tags from chunks, but a fresh pass on the summary is usually cleaner.
    
    return final_result

def _empty_result():
    return {
        "summary": "ไม่มีเนื้อหาให้วิเคราะห์ (No content to analyze)",
        "assessment": [],
        "tags": []
    }

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retrying _call_ai_api... Attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}")
)
async def _call_ai_api(text: str, is_partial: bool = False) -> dict:
    """
    Low-level function to call Typhoon AI. Includes exponential backoff for rate limits.
    """
    try:
        # Prompt adjustment for partial chunks if needed, but existing system prompt is generic enough.
        # Maybe add a note if partial? "Summarize this part of the document."
        # For now, keep it simple.

        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this content:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=32000,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        logger.info(f"AI Response: {content}")

        # Parse JSON
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            # Fallback if model returns markdown code block
            cleaned_content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned_content)

    except Exception as e:
        import traceback
        logger.error(f"AI Analysis failed: {e}")
        print("=== TYPHOON COMPLETION ERROR ===")
        print(f"Exception Type: {type(e)}")
        print(f"Error Message: {str(e)}")
        
        # Try to extract response body if it's an APIError/HTTPError
        if hasattr(e, 'response'):
            try:
                print(f"Response Body: {e.response.json()}")
            except Exception:
                print(f"Raw Response: {e.response.text}")
        elif hasattr(e, 'body'):
            print(f"Error Body: {e.body}")
            
        print("Traceback:")
        traceback.print_exc()
        print("================================")
        
        # Re-raise to trigger tenacity retry
        raise e
