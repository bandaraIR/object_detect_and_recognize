# Logic_extract_lisence_plate/ocr_reader.py
import easyocr
import cv2

class OCRReader:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def read_plate(self, image):
        """Reads text from a cropped license plate image."""
        results = self.reader.readtext(image)
        text = ""
        for (bbox, string, conf) in results:
            if conf > 0.4:  # confidence threshold
                text += string + " "
        return text.strip() if text else None