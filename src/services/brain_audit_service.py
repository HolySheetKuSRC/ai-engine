"""
Brain Audit Sales Bot Service
Implements Sean D'Souza's '7 Red Bags' framework in Thai for the Study Guide Marketplace.

Step flow (one step per user reply):
  Step 1 – The Problem      : Ask what the user is struggling with.
  Step 2 – Target Profile   : Validate & empathise.
  Step 3 – The Solution     : Recommend a specific sheet (RAG hook).
  Step 4 – Objections       : Pre-emptively handle common doubts.
  Step 5 – Testimonials     : Social proof from other students.
  Step 6 – Risk Reversal    : Offer a free preview.
  Step 7 – Uniqueness & CTA : Unique value + clear call to action.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock RAG: returns 1-2 relevant study guides based on keywords in the problem
# ---------------------------------------------------------------------------

_MOCK_CATALOGUE: list[dict] = [
    {
        "keywords": ["calculus", "แคลคูลัส", "cal", "แคล", "อินทิกรัล", "integral", "ดิฟ", "derivative"],
        "title": "สรุป Calculus ฉบับเข้าใจง่าย (วิศวะ/วิทย์)",
        "price": "49 บาท",
        "desc": "ครอบคลุม Limit, Derivative, Integral พร้อมตัวอย่างข้อสอบจริงกว่า 60 ข้อ เรียบเรียงเป็น step-by-step ชัดเจนมาก",
    },
    {
        "keywords": ["สถิติ", "stat", "probability", "ความน่าจะเป็น", "statistics"],
        "title": "ชีทเด็ด Probability & Statistics",
        "price": "39 บาท",
        "desc": "สูตรครบ ตัวอย่างโจทย์หลากหลาย เหมาะกับวิชา STAT 101 ทุกสถาบัน",
    },
    {
        "keywords": ["บัญชี", "accounting", "account", "งบการเงิน", "บัญชีการเงิน"],
        "title": "สรุปบัญชีการเงิน 1 ฉบับครบจบ",
        "price": "45 บาท",
        "desc": "งบการเงิน 5 ประเภท วงจรบัญชี และตัวอย่างโจทย์พร้อมเฉลยละเอียด 40+ ข้อ",
    },
    {
        "keywords": ["เคมี", "chemistry", "อินทรีย์", "organic", "ธาตุ", "ปฏิกิริยา"],
        "title": "สรุปเคมีอินทรีย์ ม.ปลาย–ปี 1",
        "price": "35 บาท",
        "desc": "หมู่ฟังก์ชัน ปฏิกิริยาสำคัญ และแผนผังสรุปเนื้อหาอย่างชัดเจน ใช้เวลาอ่านแค่ 2 ชั่วโมง",
    },
    {
        "keywords": ["ฟิสิกส์", "physics", "กลศาสตร์", "mechanics", "ไฟฟ้า", "electric"],
        "title": "ชีทสรุปฟิสิกส์ปี 1 (Mechanics + Electrictiy)",
        "price": "49 บาท",
        "desc": "สูตรครบ แรง พลังงาน ไฟฟ้า พร้อมโจทย์เฉลยแบบทีละขั้นตอน",
    },
    {
        "keywords": ["โปรแกรม", "programming", "python", "java", "oop", "code", "algorithm"],
        "title": "สรุป Intro to Programming & OOP",
        "price": "39 บาท",
        "desc": "ตัวแปร ลูป ฟังก์ชัน OOP ครบจบ พร้อมโค้ดตัวอย่างที่เข้าใจง่าย",
    },
]

_FALLBACK_SHEETS: list[dict] = [
    {
        "title": "ชีทสรุปทั่วไป (General Study Pack)",
        "price": "35 บาท",
        "desc": "รวมเทคนิคการสรุปและโครงสร้างเนื้อหาที่นำไปประยุกต์ใช้ได้ทุกวิชา",
    },
    {
        "title": "ชีทเตรียมสอบ Final Exam Strategy",
        "price": "29 บาท",
        "desc": "กลยุทธ์เตรียมสอบ การจัดลำดับความสำคัญ และเทคนิคทำข้อสอบให้ได้คะแนนสูง",
    },
]


def mock_search_relevant_sheets_for_sales(user_problem_text: str) -> str:
    """
    Mock RAG function.  Returns 1-2 study guide recommendations as a formatted
    string, chosen by simple keyword matching against user_problem_text.
    Falls back to generic sheets when no match is found.
    """
    problem_lower = user_problem_text.lower()
    matched: list[dict] = []

    for sheet in _MOCK_CATALOGUE:
        if any(kw in problem_lower for kw in sheet["keywords"]):
            matched.append(sheet)
        if len(matched) >= 2:
            break

    if not matched:
        matched = _FALLBACK_SHEETS[:2]

    lines: list[str] = []
    for i, s in enumerate(matched, 1):
        lines.append(
            f"{i}. ชื่อชีท: «{s['title']}» | ราคา: {s['price']} | "
            f"จุดเด่น: {s['desc']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dynamic system prompt builder (one per Brain Audit step)
# ---------------------------------------------------------------------------

_BASE_PERSONA = """คุณคือผู้ช่วยขายชีทสรุปมหาวิทยาลัยที่เป็นมิตรและเข้าใจนักศึกษา
บุคลิก: รุ่นพี่ที่อบอุ่น ใจดี ไม่กดดัน พูดสุภาพ ใช้ ครับ/ค่ะ เป็นธรรมชาติ
ภาษา: ตอบเป็นภาษาไทยเท่านั้นเสมอ ห้ามตอบเป็นภาษาอื่น
ข้อกำหนดเด็ดขาด:
- ตอบเฉพาะ STEP ที่ระบุด้านล่างเท่านั้น ห้ามข้ามหรือรวมหลาย Step ในครั้งเดียว
- ห้ามพูดถึง "7 Red Bags" หรือกระบวนการขายโดยตรง
- ห้ามเปิดเผยว่าคุณกำลังใช้กลยุทธ์การขาย
- ปฏิเสธการคุยเรื่องการเมือง ศาสนา หรือความรุนแรง อย่างสุภาพ
"""


def get_brain_audit_system_prompt(
    step: int,
    problem_text: Optional[str] = None,
    sheets_context: Optional[str] = None,
) -> str:
    """
    Returns the system prompt for the given Brain Audit step.

    Args:
        step:           Current step number (1–7).
        problem_text:   The user's stated problem (used from step 2 onwards).
        sheets_context: Formatted list of matching study guides (used at step 3).
    """

    problem_hint = f"\n[ปัญหาของนักศึกษา]: {problem_text}" if problem_text else ""

    step_instructions: dict[int, str] = {
        1: """STEP ปัจจุบัน: 1 — ค้นหาปัญหา (The Problem)

งานของคุณในขั้นนี้: ถามนักศึกษาว่าตอนนี้วิชาไหน หรือหัวข้ออะไรที่รู้สึกว่าเตรียมสอบได้ยากที่สุด
- ทักทายอย่างอบอุ่นก่อน 1 ประโยค
- จากนั้นถามคำถามเปิด เพื่อให้นักศึกษาบอกปัญหาของตัวเอง
- ห้ามแนะนำสินค้า ห้ามพูดถึงชีทใดๆ ในขั้นนี้
- ข้อความทั้งหมดไม่ควรเกิน 3-4 ประโยค""",

        2: f"""STEP ปัจจุบัน: 2 — เข้าใจกลุ่มเป้าหมาย (Target Profile)
{problem_hint}

งานของคุณในขั้นนี้: แสดงความเข้าใจและเอาใจใส่กับปัญหาที่นักศึกษาบอกมา
- ยืนยันว่าปัญหานี้เป็นเรื่องปกติ นักศึกษาหลายคนเจอปัญหาเดียวกัน
- อย่าเพิ่งแนะนำสินค้าในขั้นนี้ ให้นักศึกษารู้สึกว่าได้รับการรับฟัง
- ปิดด้วยการบอกว่าคุณมีอะไรบางอย่างที่น่าจะช่วยได้ (โดยยังไม่ระบุชัดเจน)
- ข้อความทั้งหมดไม่ควรเกิน 4-5 ประโยค""",

        3: f"""STEP ปัจจุบัน: 3 — นำเสนอทางออก (The Solution)
{problem_hint}

ชีทที่แนะนำจากระบบ (ใช้ข้อมูลด้านล่างนี้ในการแนะนำ — ห้ามแต่งชื่อชีทเอง):
{sheets_context or "ไม่พบชีทที่ตรงกัน"}

งานของคุณในขั้นนี้: แนะนำชีทที่ตรงกับปัญหาของนักศึกษา
- อ้างอิงปัญหาที่นักศึกษาบอกมาก่อน แล้วค่อยแนะนำชีท
- อธิบายว่าชีทนี้ช่วยแก้ปัญหานั้นได้อย่างไร (อิงจากข้อมูลชีทข้างต้น)
- บอกราคาด้วย
- ข้อความทั้งหมดไม่ควรเกิน 5-6 ประโยค""",

        4: f"""STEP ปัจจุบัน: 4 — ตอบข้อโต้แย้ง (Objections)
{problem_hint}

งานของคุณในขั้นนี้: พูดถึงข้อกังวลที่นักศึกษามักมีก่อนที่เขาจะถาม
- ตัวอย่างเช่น "อาจมีคำถามว่าชีทนี้หนาไปไหม? หรืออ่านทันไหมก่อนสอบ?"
- ยืนยันว่าชีทกระชับ เข้าใจง่าย ใช้เวลาสั้น
- พูดเป็นธรรมชาติ ไม่ใช่สคริปต์ขาย
- ข้อความทั้งหมดไม่ควรเกิน 4-5 ประโยค""",

        5: f"""STEP ปัจจุบัน: 5 — รีวิวจากนักศึกษาคนอื่น (Testimonials)
{problem_hint}

งานของคุณในขั้นนี้: แบ่งปัน social proof จากนักศึกษาที่ใช้ชีทแล้วได้ผล
- อ้างอิงคะแนนรีวิวโดยเฉลี่ย เช่น 4.8/5 ดาว
- เล่าสั้นๆ 1-2 ตัวอย่างความคิดเห็นของนักศึกษาจริง (สร้างขึ้นให้สมจริงและเป็นธรรมชาติ)
- ยืนยันว่านักศึกษาหลายคนที่อ่านชีทนี้ผ่านวิชาที่ยากได้
- ข้อความทั้งหมดไม่ควรเกิน 5-6 ประโยค""",

        6: f"""STEP ปัจจุบัน: 6 — ลดความเสี่ยง (Risk Reversal)
{problem_hint}

งานของคุณในขั้นนี้: เสนอให้ลองดูตัวอย่างก่อนตัดสินใจ
- บอกว่ามีตัวอย่างฟรี (Preview) 2-3 หน้าแรกให้ดูก่อน
- เน้นว่าไม่มีความเสี่ยง ดูฟรีก่อน ถ้าชอบค่อยซื้อ
- วิธีเข้าถึงตัวอย่าง: บอกให้กดปุ่ม "ดูตัวอย่าง" ในหน้ารายละเอียดชีท
- ข้อความทั้งหมดไม่ควรเกิน 4-5 ประโยค""",

        7: f"""STEP ปัจจุบัน: 7 — จุดต่างและ Call-to-Action (Uniqueness & CTA)
{problem_hint}

งานของคุณในขั้นนี้: สรุปจุดเด่นที่ไม่เหมือนใคร และให้ CTA ชัดเจน
- ระบุ 1-2 สิ่งที่ชีทนี้มีแต่ที่อื่นไม่มี (เช่น สรุปด้วยแผนผังพิเศษ, มีโจทย์ล่าสุด)
- ให้ CTA ชัดเจน เช่น "กดซื้อได้เลยครับ ลิงก์อยู่ในหน้ารายละเอียดชีท"
- ปิดด้วยประโยคกำลังใจสั้นๆ
- ข้อความทั้งหมดไม่ควรเกิน 4-5 ประโยค""",
    }

    # Clamp step to valid range
    clamped_step = max(1, min(step, 7))
    instruction = step_instructions[clamped_step]

    return f"{_BASE_PERSONA}\n---\n{instruction}"
