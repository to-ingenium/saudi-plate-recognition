from ultralytics import YOLO
from pathlib import Path

detector = YOLO("../models/detector/weights/best.pt")
results = detector("TEST ONLY\WhatsApp Image 2026-03-27 at 2.04.43 PM.jpeg", conf=0.1, save=True)

for r in results:
    print(f"Detections: {len(r.boxes)}")
    for box in r.boxes:
        print(f"  conf={float(box.conf[0]):.3f}  box={[int(x) for x in box.xyxy[0]]}")