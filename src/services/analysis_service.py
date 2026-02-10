
import os
import json
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
    Analyzes the OCR text using Typhoon AI to generate summary, assessment, and tags.
    """
    if not full_text:
        return {
            "summary": "No content to analyze.",
            "assessment": [],
            "tags": []
        }

    # Limit text length to avoid token limits (approx 6k chars to stay safe with Typhoon API)
    truncated_text = full_text[:6000]

    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this content:\n\n{truncated_text}"}
            ],
            temperature=0.3,
            max_tokens=8192,  # Typhoon SDK: max_tokens = Prompt + Response
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
