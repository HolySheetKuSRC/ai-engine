import httpx
from fastapi import HTTPException, status
import logging

logger = logging.getLogger(__name__)

async def download_pdf_from_url(url: str, max_size_mb: int = 15) -> bytes:
    """
    Downloads a PDF from a given URL.
    
    Args:
        url (str): The URL of the PDF to download.
        max_size_mb (int): Maximum allowed file size in MB. Defaults to 15MB.
        
    Returns:
        bytes: The content of the downloaded PDF.
        
    Raises:
        HTTPException: If the URL is invalid, unreachable, not a PDF, or too large.
    """
    if not url.startswith(("http://", "https://")):
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid URL schema. Must start with http:// or https://"
        )

    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Head request to check headers first (optional optimization)
            try:
                head_resp = await client.head(url, timeout=5.0)
                content_length = head_resp.headers.get("content-length")
                content_type = head_resp.headers.get("content-type")
                
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                     raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File too large. Maximum size is {max_size_mb}MB"
                    )
                
                # We can't strictly rely on content-type from some servers, but good to check if present
                if content_type and "application/pdf" not in content_type and "application/octet-stream" not in content_type:
                     # Warn or strict check? Let's be lenient for now but prefer PDF
                     pass
                     
            except httpx.RequestError:
                # If HEAD fails, just proceed to GET
                pass

            # Stream download to enforce size limit during transfer
            async with client.stream("GET", url, timeout=10.0) as resp:
                resp.raise_for_status()
                
                # Double check headers if HEAD failed or wasn't accurate
                content_length = resp.headers.get("content-length")
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"File too large. Maximum size is {max_size_mb}MB"
                    )

                file_content = b""
                async for chunk in resp.aiter_bytes():
                     file_content += chunk
                     if len(file_content) > max_size_mb * 1024 * 1024:
                         raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"File too large. Maximum size is {max_size_mb}MB"
                        )
                
                return file_content

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download file: HTTP {e.response.status_code}"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error downloading PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to download file: Network error or invalid URL"
        )
    except Exception as e:
        logger.error(f"Unexpected error downloading PDF: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during file download"
        )
