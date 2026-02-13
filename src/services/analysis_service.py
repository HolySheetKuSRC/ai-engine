import os
import json
import asyncio
from openai import AsyncOpenAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typhoon Client (OpenAI-compatible)
TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=BASE_URL
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
        
    CHUNK_SIZE = 4000
    
    # If text is small, direct analysis
    if len(full_text) <= CHUNK_SIZE:
        return await _call_ai_api(full_text)
        
    # Chunking Strategy
    chunks = [full_text[i:i+CHUNK_SIZE] for i in range(0, len(full_text), CHUNK_SIZE)]
    logger.info(f"Text too large ({len(full_text)} chars). Splitting into {len(chunks)} chunks.")
    
    # Process chunks concurrently
    tasks = [_call_ai_api(chunk, is_partial=True) for chunk in chunks]
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
        "summary": "No content to analyze.",
        "assessment": [],
        "tags": []
    }

async def _call_ai_api(text: str, is_partial: bool = False) -> dict:
    """
    Low-level function to call Typhoon AI.
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
            max_tokens=2048 if is_partial else 4096,
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
        logger.error(f"AI Analysis failed: {e}")
        return {
            "summary": "AI Analysis failed.",
            "assessment": [str(e)],
            "tags": ["#Error"]
        }
