import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
IMG_DIR = BASE_DIR / "data" / "train" / "images"
LBL_DIR = BASE_DIR / "data" / "train" / "labels"

images = list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png"))
random.shuffle(images)

to_delete = []

for img_path in images:
    lbl_path = LBL_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        continue

    img = Image.open(img_path)
    w, h = img.size

    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.imshow(img)
    ax.set_title(f"{img_path.name}\nClose window to continue", fontsize=9)
    ax.axis("off")

    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x1 = (cx - bw/2) * w
        y1 = (cy - bh/2) * h
        rect = patches.Rectangle(
            (x1, y1), bw * w, bh * h,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()  # blocks until you close the window

    answer = input("Delete this image? (y/n/q): ").strip().lower()
    if answer == "y":
        to_delete.append((img_path, lbl_path))
        print(f"Marked for deletion: {img_path.name}")
    elif answer == "q":
        break

# ── Delete all marked files ───────────────────────────────────────────────────
for img_path, lbl_path in to_delete:
    img_path.unlink()
    lbl_path.unlink()

print(f"\nDeleted {len(to_delete)} images. Done.")