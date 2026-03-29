import cv2
from pathlib import Path
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
ROOT          = BASE_DIR.parent
DETECTOR_W    = ROOT / "models"  / "detector" / "weights" / "best.pt"
READER_W      = ROOT / "models"  / "reader"   / "weights" / "best.pt"
INPUT_DIR     = ROOT / "assets"  / "test_samples"
OUTPUT_DIR    = ROOT / "assets"  / "test_samples" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
detector = YOLO(str(DETECTOR_W))
reader   = YOLO(str(READER_W))
print("Models ready.\n")

# ── Collect input files ───────────────────────────────────────────────────────
input_files = [
    p for p in INPUT_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS
    and "output" not in p.parts
]

if not input_files:
    print(f"No image or video files found in {INPUT_DIR}")
    exit()

print(f"Found {len(input_files)} file(s):")
for f in input_files:
    print(f"  {f.name}")
print()

# ── Helper: detect plates in a frame, return (annotated_frame, detected) ─────
def process_frame(frame):
    det_results = detector(frame, verbose=False)[0]
    detected = False

    for box in det_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        if conf < 0.25:
            continue
        detected = True

        # Crop + resize for reader
        plate_crop = frame[y1:y2, x1:x2]
        if plate_crop.size == 0:
            continue
        plate_crop = cv2.resize(plate_crop, (320, 160))

        # Read characters
        read_results = reader(plate_crop, verbose=False)[0]
        characters = []
        for r_box in read_results.boxes:
            cx    = float(r_box.xyxy[0][0])
            label = reader.names[int(r_box.cls[0])]
            characters.append((cx, label))
        characters.sort(key=lambda x: x[0])
        plate_text = "".join(c[1] for c in characters) if characters else "?"

        # Draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_str = f"{plate_text}  {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, label_str, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return frame, detected

# ── Process files ─────────────────────────────────────────────────────────────
for file_path in input_files:
    ext = file_path.suffix.lower()
    print(f"Processing: {file_path.name}")

    # ── Image ─────────────────────────────────────────────────────────────────
    if ext in IMAGE_EXTS:
        frame = cv2.imread(str(file_path))
        if frame is None:
            print(f"  Could not read, skipping.")
            continue
        annotated, detected = process_frame(frame)
        out_path = OUTPUT_DIR / file_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  {'Plate detected' if detected else 'No plate found'} → output/{out_path.name}")

    # ── Video — save only frames where a plate is detected ────────────────────
    elif ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            print(f"  Could not open, skipping.")
            continue

        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved      = 0
        frame_idx  = 0
        out_folder = OUTPUT_DIR / file_path.stem
        out_folder.mkdir(exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            annotated, detected = process_frame(frame)

            if detected:
                out_path = out_folder / f"frame_{frame_idx:05d}.jpg"
                cv2.imwrite(str(out_path), annotated)
                saved += 1

            if frame_idx % 10 == 0:
                print(f"  Frame {frame_idx}/{total} — {saved} saved", end="\r")

        cap.release()
        print(f"  Done — {saved} detected frames saved to output/{out_folder.name}/")

print(f"\nAll done. Outputs in: {OUTPUT_DIR}")