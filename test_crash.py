import asyncio
from src.schemas.chat import ChatRequest
from src.services.chat_service import process_chat
from src.database import async_session_maker

async def main():
    req = ChatRequest(session_id="dev-test", message="hello", sheet_id=None)
    async with async_session_maker() as db:
        res = await process_chat(req, db)
        print("Success:", res)

if __name__ == "__main__":
    asyncio.run(main())
