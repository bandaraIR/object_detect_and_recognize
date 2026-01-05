# app.py — Object Detection + Full License Plate Recognition (Live Webcam) + Firestore

import cv2
import easyocr
from ultralytics import YOLO
import os
import numpy as np
from firebase_admin import firestore
from firebase_config import db  # Firestore client

# --- Model paths ---
OBJECT_MODEL_PATH = "yolov8n.pt"  # Detects cars, bikes, persons
PLATE_MODEL_PATH = "/Users/imal/Downloads/object_detect_project/Object_detection_plates_lisence/model_yolo/results_version_3/content/runs/detect/train/best (1).pt"

# --- Load models ---
if not os.path.exists(OBJECT_MODEL_PATH):
    raise FileNotFoundError(f"Object model not found: {OBJECT_MODEL_PATH}")
object_model = YOLO(OBJECT_MODEL_PATH)
print("✅ YOLOv8 object detector loaded.")

if not os.path.exists(PLATE_MODEL_PATH):
    raise FileNotFoundError(f"Plate model not found: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)
print("✅ YOLOv8 plate detector loaded.")

# --- Initialize EasyOCR ---
ocr_reader = easyocr.Reader(['en'])
print("✅ EasyOCR loaded.")

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("ERROR: Could not open webcam.")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_count = 0
print("🎥 Webcam started. Press 'q' to exit.")

# --- Stop line position ---
STOP_LINE_Y = 350  # Adjust depending on camera angle, moved slightly down

# --- Horizontal merge function ---
def merge_boxes_horizontally(boxes, max_gap=15):
    """Merge boxes that are close horizontally."""
    if len(boxes) == 0:
        return []
    boxes = sorted(boxes, key=lambda b: b[0])
    merged = []
    current = boxes[0]
    for b in boxes[1:]:
        if b[0] - current[2] <= max_gap:
            current = [min(current[0], b[0]),
                       min(current[1], b[1]),
                       max(current[2], b[2]),
                       max(current[3], b[3])]
        else:
            merged.append(current)
            current = b
    merged.append(current)
    return merged

# --- Preprocessing for OCR ---
def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h > 0 and w > 0:
        target_height = 100
        scale = target_height / h
        new_width = int(w * scale)
        gray = cv2.resize(gray, (new_width, target_height))
    gray = cv2.medianBlur(gray, 3)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray

# --- Save plate to Firestore ---
def save_plate_to_firestore(plate_number):
    if plate_number == "UNREADABLE":
        return
    doc_ref = db.collection("violations").document()
    doc_ref.set({
        "plate_number": plate_number,
        "timestamp": firestore.SERVER_TIMESTAMP
    })
    print(f"✅ Saved to Firestore: {plate_number}")

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    frame_count += 1

    # 🟢 Draw stop line
    cv2.line(frame, (0, STOP_LINE_Y), (frame.shape[1], STOP_LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, "STOP LINE", (10, STOP_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 1️⃣ Detect general objects
    obj_results = object_model.predict(frame, conf=0.25, save=False, verbose=False)
    for box, cls, conf in zip(obj_results[0].boxes.xyxy, obj_results[0].boxes.cls, obj_results[0].boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        label = object_model.model.names[int(cls)]
        color = (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2️⃣ Detect number plates
    plate_results = plate_model.predict(frame, conf=0.3, save=False, verbose=False)
    boxes = [list(map(int, b)) for b in plate_results[0].boxes.xyxy]
    merged_boxes = merge_boxes_horizontally(boxes)

    for x1, y1, x2, y2 in merged_boxes:
        padding = 5
        x1_pad = max(0, x1-padding)
        y1_pad = max(0, y1-padding)
        x2_pad = min(frame.shape[1], x2+padding)
        y2_pad = min(frame.shape[0], y2+padding)
        plate_crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if plate_crop.size == 0:
            continue

        processed_plate = preprocess_plate(plate_crop)

        # OCR with allowlist
        ocr_result = ocr_reader.readtext(
            processed_plate,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            detail=0
        )
        full_plate = ''.join(ocr_result).upper().replace(' ', '') if ocr_result else "UNREADABLE"

        # Draw box and label
        color = (0, 255, 0) if full_plate != "UNREADABLE" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, f"Plate: {full_plate}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # --- Only store plates below the stop line ---
        plate_center_y = (y1 + y2) // 2
        if plate_center_y >= STOP_LINE_Y:
            print(f"Frame {frame_count}: Number Plate - {full_plate} ✅ Stored")
            save_plate_to_firestore(full_plate)
        else:
            print(f"Frame {frame_count}: Number Plate - {full_plate} ❌ Not Stored")

    # Show frame
    cv2.imshow("Objects + Plate Recognition", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("🎥 Webcam released. Goodbye!")