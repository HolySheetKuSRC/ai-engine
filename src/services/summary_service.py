import os
from openai import AsyncOpenAI

TYPHOON_API_KEY = os.getenv("TYPHOON_API_KEY")
TYPHOON_BASE_URL = "https://api.opentyphoon.ai/v1"

client = AsyncOpenAI(
    api_key=TYPHOON_API_KEY,
    base_url=TYPHOON_BASE_URL
)

SYSTEM_PROMPT = """คุณคือผู้ช่วยสรุปเลคเชอร์ สรุปเนื้อหาเลคเชอร์ดังกล่าวโดยโฟกัสแค่เสียงอาจารย์ สรุปเนื้อหาอิงตามเสียง สรุปแบบกลับมาภาษาไทยอ่านแล้วเข้าใจได้ง่าย layoutจัดดีๆ (ใช้ Markdown, Bullet points, และแบ่งหัวข้อให้ชัดเจน)"""

async def summarize_lecture(text: str) -> str:
    """
    Summarize lecture transcript using Typhoon LLM.
    
    Args:
        text (str): The raw transcript text.
        
    Returns:
        str: The summary text.
    """
    response = await client.chat.completions.create(
        model="typhoon-v2.5-30b-a3b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0.7,
        max_tokens=32000,
    )
    
    return response.choices[0].message.content
