import os
import shutil
import random
import zipfile
import yaml
from pathlib import Path

import albumentations as A
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data"
ARCHIVE      = DATA_DIR / "archive.zip"
EXTRACTED    = DATA_DIR / "saudi-license-plate-characters"
SOURCE_ROOT  = EXTRACTED / "License-Characters-by-2-27classes"
SOURCE_TRAIN = SOURCE_ROOT / "train"
SOURCE_TEST  = SOURCE_ROOT / "test"
DEST         = BASE_DIR / "dataset"
VAL_SPLIT    = 0.10

# ── Unzip ─────────────────────────────────────────────────────────────────────
with zipfile.ZipFile(ARCHIVE, "r") as zf:
    zf.extractall(DATA_DIR)

# ── Build split folders ───────────────────────────────────────────────────────
for split in ["train", "val", "test"]:
    (DEST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DEST / "labels" / split).mkdir(parents=True, exist_ok=True)

def copy_files(src_dir, split):
    for fname in os.listdir(src_dir):
        src = Path(src_dir) / fname
        if fname.endswith((".jpeg", ".jpg", ".png")):
            shutil.copy(src, DEST / "images" / split / fname)
        elif fname.endswith(".txt"):
            shutil.copy(src, DEST / "labels" / split / fname)

copy_files(SOURCE_TEST, "test")

all_images = [f for f in os.listdir(SOURCE_TRAIN) if f.endswith((".jpeg", ".jpg", ".png"))]
random.seed(42)
random.shuffle(all_images)

val_images = set(all_images[:int(len(all_images) * VAL_SPLIT)])

for fname in os.listdir(SOURCE_TRAIN):
    src  = SOURCE_TRAIN / fname
    stem = os.path.splitext(fname)[0]

    if fname.endswith((".jpeg", ".jpg", ".png")):
        split = "val" if fname in val_images else "train"
        shutil.copy(src, DEST / "images" / split / fname)
    elif fname.endswith((".txt", ".xml")):
        img_name = next((img for img in all_images if os.path.splitext(img)[0] == stem), None)
        if img_name:
            split = "val" if img_name in val_images else "train"
            shutil.copy(src, DEST / "labels" / split / fname)

print(f"Train: {len(list((DEST / 'images' / 'train').iterdir()))} images")
print(f"Val:   {len(list((DEST / 'images' / 'val').iterdir()))} images")
print(f"Test:  {len(list((DEST / 'images' / 'test').iterdir()))} images")

# ── data.yaml ─────────────────────────────────────────────────────────────────
class_map = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "D", 13: "E", 14: "G",
    15: "H", 16: "J", 17: "K", 18: "L", 19: "N",
    20: "R", 21: "S", 22: "T", 23: "U", 24: "V",
    25: "X", 26: "Z"
}

yaml_path = DEST / "data.yaml"
with open(yaml_path, "w") as f:
    yaml.dump({
        "path":  str(DEST),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    27,
        "names": class_map,
    }, f, sort_keys=True, allow_unicode=True)

# ── Train ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    augmentations = [
        A.BBoxSafeRandomCrop(p=0.5),
        A.Resize(640, 640),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.25),
                            hole_width_range=(0.1, 0.25), p=1.0),
            A.GridDropout(ratio=0.5, unit_size_range=(10, 20), p=1.0)
        ], p=0.5),
    ]

    model = YOLO("yolov8s.pt")
    model.train(
        data        = str(yaml_path),
        epochs      = 20,
        imgsz       = 640,
        batch       = 16,
        project     = str(BASE_DIR),
        name        = "train",
        exist_ok    = True,
        augmentations = augmentations,
    )

    model.export(format="onnx")
    print(f"\nWeights saved to: {BASE_DIR / 'train' / 'weights'}")
