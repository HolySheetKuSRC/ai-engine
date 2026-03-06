from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, or_
from openai import AsyncOpenAI
from src.models.chat import ChatHistory
from src.models.ai_dataset import AiDatasetRecord # RAG source
from src.database import AnalyzeJob  # Analysis result source
from src.schemas.chat import ChatRequest
from src.config import settings

from langsmith import traceable
from langsmith.wrappers import wrap_openai
import httpx
from supabase import create_client, Client
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Typhoon Client
client = wrap_openai(AsyncOpenAI(
    api_key=settings.TYPHOON_API_KEY,
    base_url=settings.TYPHOON_BASE_URL
))

# Initialize Supabase Client for Vector Search
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def get_gemini_embedding(text: str) -> list[float]:
    """Get embedding vector using Google Gemini API. Handles 404 gracefully."""
    # Using models/embedding-001 as fallback
    url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={settings.GEMINI_API_KEY}"
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    async with httpx.AsyncClient() as gemini_http_client:
        try:
            response = await gemini_http_client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("embedding", {}).get("values", [])
        except httpx.HTTPStatusError as e:
            logger.warning(f"Gemini API Error: {e.response.status_code} - Gracefully bypassing cache.")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error getting Gemini Embedding: {e} - Bypassing cache.")
            return []


async def search_relevant_sheets(db: AsyncSession, user_message: str) -> str:
    """Search for relevant sheets based on user keywords."""
    words = [w.strip() for w in user_message.split() if len(w.strip()) > 2]
    
    if not words:
        # Return top 5 recent if no meaningful keywords
        stmt = select(AiDatasetRecord).order_by(desc(AiDatasetRecord.created_at)).limit(5)
    else:
        conditions = []
        for word in words:
            conditions.append(AiDatasetRecord.filename.ilike(f"%{word}%"))
            # handle NULL summary_text implicitly with ilike
            conditions.append(AiDatasetRecord.summary_text.ilike(f"%{word}%"))
        
        stmt = select(AiDatasetRecord).where(or_(*conditions)).limit(5)
        
    result = await db.execute(stmt)
    sheets = result.scalars().all()
    
    if not sheets:
        logger.info(f"search_relevant_sheets: No sheets found for keywords: {words}")
        return "ไม่มีข้อมูลชีทในระบบที่ตรงกับคำค้นหา"
        
    formatted_sheets = []
    for sheet in sheets:
        summary = sheet.summary_text[:100] + "..." if sheet.summary_text else "N/A"
        formatted_sheets.append(f"ID: {sheet.id} | ชื่อไฟล์: {sheet.filename} | รายละเอียด: {summary} | Tags: {sheet.tags}")
    
    logger.info(f"search_relevant_sheets: Found {len(sheets)} sheets. Matches: {formatted_sheets}")
    return "\n".join(formatted_sheets)

async def get_sales_assistant_prompt(db: AsyncSession, user_message: str) -> str:
    available_sheets_context = await search_relevant_sheets(db, user_message)
    return f"""คุณคือผู้ช่วยแนะนำชีทเรียนของแพลตฟอร์ม Study Guide Marketplace (ระดับมหาวิทยาลัย)
เป้าหมายหลัก: วิเคราะห์ความต้องการของผู้ใช้ และแนะนำชีทเรียนที่ตรงกับความต้องการมากที่สุดจาก "รายชื่อชีทที่มีในระบบ" ด้านล่างนี้เท่านั้น

[รายชื่อชีทที่มีในระบบตอนนี้]
{available_sheets_context}

กฎการแนะนำ (สำคัญมาก):
1. ความหน้าเชื่อถือ (Confidence): หากพบชีทใน "รายชื่อชีทที่มีในระบบตอนนี้" ที่เนื้อหา (รายละเอียด/Tags) ใกล้เคียงกับสิ่งที่ผู้ใช้ถาม **คุณต้องแนะนำชีทนั้นทันที** 
2. การนำเสนอ: ให้พิมพ์ชื่อชีทพร้อม จุดเด่น (อิงจากรายละเอียดและ Tags) และเน้นย้ำว่าชีทนี้ตรงกับสิ่งที่ผู้ใช้กำลังมองหา
3. หากไม่มีชีทไหนในระบบที่ใกล้เคียงคำถามเลย (หรือรายชื่อชีทว่างเปล่า) ให้ตอบสุภาพว่า "ตอนนี้ยังไม่มีชีทวิชานี้ในระบบครับ แต่ประเด็นนี้..." (แล้วจึงให้ข้อมูลทั่วไปสั้นๆ)
4. ห้ามแต่งชื่อชีท หรือแนะนำชีทที่ไม่มีอยู่ใน [รายชื่อชีทที่มีในระบบตอนนี้] เด็ดขาด (No Hallucination).
5. ห้ามแจกสูตรฟรีทั้งหมด หรือสอนเนื้อหาแบบละเอียดเกินไปจนผู้ใช้ไม่จำเป็นต้องซื้อชีท
6. ปฏิเสธการคุยเรื่องการเมือง ศาสนา ความรุนแรง อย่างสุภาพ
"""


# @traceable removed to prevent silent RecursionError crashes when serializing `db: AsyncSession`
async def process_chat(request: ChatRequest, db: AsyncSession):
    session_id = request.session_id
    user_message = request.message
    sheet_id = request.sheet_id

    # Handle invalid sheet_id inputs like "string", "", "null"
    if isinstance(sheet_id, str):
        cleaned_id = sheet_id.strip().lower()
        if cleaned_id in ["string", "null", "none", ""]:
            sheet_id = None

    # 1. Fetch History (Last 20 messages)
    stmt = (
        select(ChatHistory)
        .where(ChatHistory.session_id == session_id)
        .order_by(desc(ChatHistory.created_at))
        .limit(20)
    )
    result = await db.execute(stmt)
    history_records = result.scalars().all()
    # Reverse to chronological order for the model
    history_messages = [{"role": h.role, "content": h.content} for h in reversed(history_records)]

    # 2. Determine System Prompt (RAG or General)
    system_instruction = ""
    if sheet_id:
        # RAG Mode: Fetch context from AiDatasetRecord
        # Assuming sheet_id corresponds to id or filename in AiDatasetRecord.
        # User said "sheet_id: String (Nullable, indicates if the chat is tied to a specific study guide)"
        # But AiDatasetRecord.id is Integer.
        # Let's try to match schema. If sheet_id is passed as string '123', we cast to int.
        # If it fails, fallback or error? User requirement says "Query the AiDatasetRecord".
        
        try:
            # Query AnalyzeJob using sheet_id (which is a string like "bio" or "123")
            job_result = {}
            if sheet_id:
                stmt_rag = (
                    select(AnalyzeJob)
                    .where(AnalyzeJob.sheet_id == str(sheet_id))
                    .order_by(AnalyzeJob.created_at.desc())
                    .limit(1)
                )
                result_rag = await db.execute(stmt_rag)
                job = result_rag.scalar_one_or_none()
                if job and job.result:
                    job_result = job.result

            import json

            # 1. Safely parse JSON if SQLite returned a string
            if isinstance(job_result, str):
                try:
                    job_result = json.loads(job_result)
                except json.JSONDecodeError:
                    job_result = {}

            # 2. Extract content handling both Legacy (String) and New (List) formats
            context_text = ""
            if job_result and isinstance(job_result, dict):
                ocr_data = job_result.get("ocr_content", "")

                if isinstance(ocr_data, list):
                    # New format: List of Bounding Box dicts
                    text_blocks = []
                    for item in ocr_data:
                        if isinstance(item, dict) and "text" in item:
                            text_blocks.append(str(item["text"]))
                    context_text = "\n".join(text_blocks)
                elif isinstance(ocr_data, str):
                    # Legacy format: Raw string
                    context_text = ocr_data

                # Fallback to summary if ocr_content is completely empty
                if not context_text.strip():
                    context_text = job_result.get("summary", "")

            # 3. Validation & Debug Logging (The Ultimate Move)
            print(f"========== DEBUG CHAT [Sheet: {sheet_id}] ==========")
            print(f"Raw Job Result Type: {type(job_result)}")
            print(f"Extracted Context Length: {len(context_text)}")
            print(f"Raw Job Result Sample (first 300 chars): {str(job_result)[:300]}")
            print(f"==================================================")

            # 4. Clean out figure/image descriptions generated by OCR
            context_text = re.sub(r'<figure>.*?</figure>', '', context_text, flags=re.DOTALL)

            # 5. Clean up excess blank lines to save tokens
            context_text = re.sub(r'\n{3,}', '\n\n', context_text).strip()

            # SECURE TRUNCATION FOR LLM LIMITS 
            # Math Hack for API rules: Max prompt tokens ~8000, max_tokens=32000. Sum <= 40000.
            MAX_CHARS = 15000 
            if len(context_text) > MAX_CHARS:
                context_text = context_text[:MAX_CHARS] + "\n\n...[เนื้อหาบางส่วนถูกตัดออก]..."

            if not context_text.strip():
                 # Strictly handle direct sheet_id inquiries
                 return {
                     "session_id": session_id,
                     "message": "ขออภัยครับ ไม่พบข้อมูลเนื้อหาของชีทที่คุณระบุ หรือคุณอาจจะระบุรหัสเอกสารไม่ถูกต้อง",
                     "sheet_id": sheet_id,
                     "logs": {"error": "Sheet content not found"}
                 }

            system_instruction = f"คุณคือผู้ช่วยตอบคำถามจากเอกสารสรุปการเรียน จงตอบคำถามโดยอ้างอิงจากเนื้อหาต่อไปนี้เท่านั้น ห้ามบรรยายรูปภาพหรือกราฟิกเด็ดขาด ให้โฟกัสที่การสรุปเนื้อหาและตอบคำถามจากข้อความ (Text) เท่านั้น:\n\n{context_text}"
        except Exception as e:
            logger.error(f"Error executing RAG context generation: {e}")
            return {
                "session_id": session_id,
                "message": "เกิดข้อผิดพลาดในการโหลดข้อมูลเอกสาร",
                "sheet_id": sheet_id,
                "logs": {"error": str(e)}
            }
    else:
        # General Mode
        system_instruction = await get_sales_assistant_prompt(db, user_message)

    # 3. Handle Semantic Caching (Only for Tutor Mode)
    if sheet_id is not None:
        try:
            query_embedding = await get_gemini_embedding(user_message)
            
            # Pre-initialize cache response to avoid UnboundLocalError
            cache_response = None
            
            # Sub-check if embedding was successfully retrieved
            if query_embedding:
                # Query Supabase RPC fn
                cache_response = supabase.rpc("match_semantic_cache", {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.85,
                    "match_count": 1
                }).execute()
            
            matches = cache_response.data if cache_response else None
            if matches and len(matches) > 0:
                cached_answer = matches[0].get("response")
                if cached_answer:
                    # Cache Hit!
                    # Save User message
                    user_msg_db = ChatHistory(
                        session_id=session_id,
                        sheet_id=sheet_id,
                        role="user",
                        content=user_message
                    )
                    db.add(user_msg_db)
                    await db.commit()
                    
                    # Save Assistant cached message
                    assistant_msg_db = ChatHistory(
                        session_id=session_id,
                        sheet_id=sheet_id,
                        role="assistant",
                        content=cached_answer
                    )
                    db.add(assistant_msg_db)
                    await db.commit()
                    
                    return {
                        "session_id": session_id,
                        "message": cached_answer,
                        "sheet_id": sheet_id,
                        "logs": {"cache_hit": True}
                    }
        except Exception as e:
            logger.error(f"Semantic Cache Error: {e}")
            # Proceed to LLM Call if cache fails
            pass

    if sheet_id:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_message}
        ]
    else:
        messages = [{"role": "system", "content": system_instruction}] + history_messages + [{"role": "user", "content": user_message}]

    # 4. Save User Message
    user_msg_db = ChatHistory(
        session_id=session_id,
        sheet_id=sheet_id,
        role="user",
        content=user_message
    )
    db.add(user_msg_db)
    await db.commit()

    try:
        response = await client.chat.completions.create(
            model="typhoon-v2.5-30b-a3b-instruct",
            messages=messages,
            stream=False,
            max_tokens=32000,
            temperature=0.6,
        )
        
        full_response_text = response.choices[0].message.content
        
        # 5. Save Assistant Message
        if full_response_text:
             assistant_msg_db = ChatHistory(
                session_id=session_id,
                sheet_id=sheet_id,
                role="assistant",
                content=full_response_text
            )
             db.add(assistant_msg_db)
             await db.commit()
             
             # Save to Semantic Cache (Only for Tutor Mode)
             if sheet_id is not None:
                 try:
                     query_emb = await get_gemini_embedding(user_message)
                     if query_emb:
                         supabase.table("semantic_cache").insert({
                             "prompt": user_message,
                             "response": full_response_text,
                             "embedding": query_emb
                         }).execute()
                 except Exception as e:
                     logger.warning(f"Failed to save semantic cache: {e}")

        return {
            "session_id": session_id,
            "message": full_response_text,
            "sheet_id": sheet_id,
            "logs": {"cache_hit": False, "model": "typhoon-v2.5-30b-a3b-instruct"}
        }
        
    except Exception as e:
        return {
            "session_id": session_id,
            "message": f"Error: {str(e)}",
            "sheet_id": sheet_id,
            "logs": {"error": True}
        }

