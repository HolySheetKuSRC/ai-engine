
import httpx
import logging

logger = logging.getLogger(__name__)

async def send_webhook(webhook_url: str, data: dict):
    """
    Sends the analysis result to the configured webhook URL.
    This is a fire-and-forget operation; we log errors but don't stop the flow.
    """
    if not webhook_url:
        return

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(webhook_url, json=data, timeout=10.0)
            resp.raise_for_status()
            logger.info(f"Webhook sent successfully to {webhook_url}: {resp.status_code}")
    except Exception as e:
        logger.error(f"Failed to send webhook to {webhook_url}: {e}")
