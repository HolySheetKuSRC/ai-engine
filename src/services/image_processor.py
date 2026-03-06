import cv2
import numpy as np

def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocesses the image for better OCR and Layout Analysis.
    Returns:
        tuple: (deskewed_color_image, deskewed_binary_image)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Adaptive Thresholding (Binarization) to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
    )
    
    # Deskewing
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle based on minAreaRect behavior
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = binary.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Deskew binary map
        deskewed_binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Deskew original image
        if len(image.shape) == 3:
            deskewed_color = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else:
            deskewed_color = deskewed_binary.copy()
            
        return deskewed_color, deskewed_binary
            
    return image, binary

def detect_text_blocks(binary_image: np.ndarray) -> list[dict]:
    """
    Pass 1: Detect paragraph-level text blocks using morphological operations.
    Returns:
        list[dict]: A list of bounding box coordinates {"x": ..., "y": ..., "w": ..., "h": ...}
    """
    # Create a rectangular kernel for dilation to group text characters/lines into blocks
    # Width is large to combine horizontal words, height is large to combine lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 40)) 
    dilated = cv2.dilate(binary_image, kernel, iterations=1)
    
    # Optional: Apply morphological close to ensure solid blocks
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Filter out very small noise blocks (e.g. smaller than 30x15)
        if w > 30 and h > 15:
            blocks.append({"x": x, "y": y, "w": w, "h": h})
            
    # Sort blocks top-to-bottom, then loosely left-to-right
    blocks.sort(key=lambda b: (b['y'] // 20, b['x']))
    return blocks
