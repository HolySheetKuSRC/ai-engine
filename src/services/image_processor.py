import cv2
import numpy as np
from typing import List, Dict, Tuple

def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Calculates the skew angle of the text and returns a straightened image.
    Works best on grayscale or BGR images.
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Invert and threshold to find text pixels
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # Get all non-black pixels
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) == 0:
        return image
        
    # Calculate the minimum bounding rectangle of all text pixels
    angle = cv2.minAreaRect(coords)[-1]
    
    # Adjust the angle correctly
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    # Ignore extremely small skew angles
    if abs(angle) < 0.5:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Rotate the image
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Determine border properties depending on image channels
    if len(image.shape) == 3:
        borderValue = (255, 255, 255) # White for BGR
    else:
        borderValue = 255 # White for grayscale
        
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=borderValue)
    return rotated

def preprocess_for_layout_analysis(image: np.ndarray) -> np.ndarray:
    """
    Takes an aligned/deskewed image and produces a binary mask
    optimized for layout analysis (contour finding).
    Returns a binary image where text/foreground is WHITE (255) and background is BLACK (0).
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # Adaptive binarization to handle varied lighting
    # We use THRESH_BINARY_INV so text becomes white
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 9
    )
    
    # Optional: Remove horizontal lines if they connect columns/rows
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    remove_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # Subtract to isolate text further, or just find contours of the lines to paint them black
    cnts, _ = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        # Paint the horizontal lines black on the binary mask
        cv2.drawContours(binary, [c], -1, 0, -2) # -2 thickness means filled slightly thicker

    return binary

def detect_text_blocks(binary_image: np.ndarray) -> List[Dict[str, int]]:
    """
    Takes a binary image (text=255, bg=0) and returns a list of paragraph-level bounding boxes.
    """
    # Dilate heavily to merge words into lines, and lines into blocks/paragraphs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    dilated = cv2.dilate(binary_image, kernel, iterations=3)

    # Find contours on the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blocks = []
    min_area = 400 # Filter out noise

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:
            # We want to give a slight padding around the bounding box to not cut off text
            pad_x = 5
            pad_y = 5
            
            # Ensure coordinates are within image boundaries
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(binary_image.shape[1], x + w + pad_x)
            y2 = min(binary_image.shape[0], y + h + pad_y)
            
            blocks.append({
                "x": x1, 
                "y": y1, 
                "w": x2 - x1, 
                "h": y2 - y1
            })
            
    # Sort blocks top-to-bottom, left-to-right
    # A tolerance grouping for Y (e.g. 20 pixels) helps sort properly on roughly same lines
    blocks.sort(key=lambda b: (b["y"] // 20, b["x"]))
            
    return blocks
