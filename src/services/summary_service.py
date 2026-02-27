import os
import asyncio
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL,
    timeout=45.0,
    max_retries=1
)

SYSTEM_PROMPT = """คุณคือผู้ช่วยสรุปเลคเชอร์ สรุปเนื้อหาเลคเชอร์ดังกล่าวโดยโฟกัสแค่เสียงอาจารย์ สรุปเนื้อหาอิงตามเสียง สรุปแบบกลับมาภาษาไทยอ่านแล้วเข้าใจได้ง่าย layoutจัดดีๆ (ใช้ Markdown, Bullet points, และแบ่งหัวข้อให้ชัดเจน)"""
MODEL_NAME = "typhoon-v2.5-30b-a3b-instruct"

async def summarize_lecture(text: str) -> str:
    """
    Summarize lecture transcript using Typhoon LLM.
    Handles chunking if the transcript is too large.
    
    Args:
        text (str): The raw transcript text.
        
    Returns:
        str: The summary text.
    """
    if not text or not text.strip():
        return "ไม่มีเสียงพูดในไฟล์นี้ (No speech detected)"

    CHUNK_SIZE = 15000
    
    # If text is small, direct analysis
    if len(text) <= CHUNK_SIZE:
        return await _call_typhoon_summarize(text)
        
    # Chunking Strategy
    chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    logger.info(f"Transcript too large ({len(text)} chars). Splitting into {len(chunks)} chunks.")
    
    semaphore = asyncio.Semaphore(3)
    
    async def process_chunk_with_limit(chunk: str) -> str:
        async with semaphore:
            await asyncio.sleep(2)  # Polite delay
            return await _call_typhoon_summarize(chunk, is_partial=True)

    tasks = [process_chunk_with_limit(chunk) for chunk in chunks]
    results = await asyncio.gather(*tasks)
    
    # Final Pass
    combined_summary = "\n\n".join(results)
    logger.info("Running final summary on merged summaries...")
    final_summary = await _call_typhoon_summarize(combined_summary)
    
    return final_summary

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=5, min=5, max=30),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: logger.warning(f"Retrying _call_typhoon_summarize... Attempt {retry_state.attempt_number} after error: {retry_state.outcome.exception()}")
)
async def _call_typhoon_summarize(content: str, is_partial: bool = False) -> str:
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.7,
            max_tokens=32000,
        )
        return response.choices[0].message.content
        
    except Exception as e:
        import traceback
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
        raise e
