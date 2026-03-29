import cv2
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import albumentations as A
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent          # 1stModel_plate_detector/
DATA_YAML   = BASE_DIR / "data" / "data.yaml"
WEIGHTS_DIR = BASE_DIR / "weights"
TRAIN_DIR   = BASE_DIR / "train"

if __name__ == "__main__":
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("=" * 50)
    print("  Training plate detector")
    print("=" * 50)

    model = YOLO("yolov8s.pt")

    augmentations = [
        A.BBoxSafeRandomCrop(p=0.5),
        A.Resize(640, 640),
        A.RandomBrightnessContrast(p=0.2),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(0.1, 0.25),
                            hole_width_range=(0.1, 0.25), p=1.0),
            A.GridDropout(ratio=0.5, unit_size_range=(10, 20), p=1.0)
        ], p=0.5)
    ]

    model.train(
        data         = str(DATA_YAML),
        epochs       = 50,
        imgsz        = 640,
        batch        = 16,
        workers      = 2,
        patience     = 10,
        project      = str(BASE_DIR),
        name         = "train",
        exist_ok     = True,
        augmentations= augmentations,
    )

    # ── Copy best weights ─────────────────────────────────────────────────────
    best_src = TRAIN_DIR / "weights" / "best.pt"
    last_src = TRAIN_DIR / "weights" / "last.pt"

    if best_src.exists():
        shutil.copy(best_src, WEIGHTS_DIR / "best.pt")
        shutil.copy(last_src, WEIGHTS_DIR / "last.pt")
        print(f"\nWeights saved to {WEIGHTS_DIR}")

    # ── Load results.csv ──────────────────────────────────────────────────────
    csv_path = TRAIN_DIR / "results.csv"
    if not csv_path.exists():
        print("results.csv not found, skipping plots.")
    else:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        epochs = df["epoch"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Plate Detector — Training Results", fontsize=15, fontweight="bold")

        # Box loss
        ax = axes[0, 0]
        ax.plot(epochs, df["train/box_loss"], label="Train", color="#2196F3")
        ax.plot(epochs, df["val/box_loss"],   label="Val",   color="#FF5722")
        ax.set_title("Box Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Class loss
        ax = axes[0, 1]
        ax.plot(epochs, df["train/cls_loss"], label="Train", color="#2196F3")
        ax.plot(epochs, df["val/cls_loss"],   label="Val",   color="#FF5722")
        ax.set_title("Class Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # mAP
        ax = axes[1, 0]
        ax.plot(epochs, df["metrics/mAP50(B)"],    label="mAP50",    color="#4CAF50")
        ax.plot(epochs, df["metrics/mAP50-95(B)"], label="mAP50-95", color="#9C27B0")
        ax.set_title("mAP")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Precision & Recall
        ax = axes[1, 1]
        ax.plot(epochs, df["metrics/precision(B)"], label="Precision", color="#FF9800")
        ax.plot(epochs, df["metrics/recall(B)"],    label="Recall",    color="#00BCD4")
        ax.set_title("Precision & Recall")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = BASE_DIR / "training_results.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to {plot_path}")
        plt.show()

        # Print final metrics
        last = df.iloc[-1]
        best_map_idx = df["metrics/mAP50(B)"].idxmax()
        best = df.iloc[best_map_idx]

        print("\n── Final epoch ──────────────────────────────")
        print(f"  mAP50:      {last['metrics/mAP50(B)']:.4f}")
        print(f"  mAP50-95:   {last['metrics/mAP50-95(B)']:.4f}")
        print(f"  Precision:  {last['metrics/precision(B)']:.4f}")
        print(f"  Recall:     {last['metrics/recall(B)']:.4f}")
        print(f"\n── Best epoch ({int(best['epoch'])}) ──────────────────────────")
        print(f"  mAP50:      {best['metrics/mAP50(B)']:.4f}")
        print(f"  mAP50-95:   {best['metrics/mAP50-95(B)']:.4f}")

    # ── Sample test on 6 random test images ──────────────────────────────────
    print("\n── Running sample test ──────────────────────────")

    best_model   = YOLO(str(WEIGHTS_DIR / "best.pt"))
    test_img_dir = BASE_DIR / "data" / "test" / "images"
    test_images  = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))

    if len(test_images) == 0:
        print("No test images found.")
    else:
        samples = random.sample(test_images, min(6, len(test_images)))

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle("Sample Detections on Test Images", fontsize=14, fontweight="bold")
        axes = axes.flatten()

        for i, img_path in enumerate(samples):
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            preds   = best_model(img_bgr, verbose=False)[0]

            ax = axes[i]
            ax.imshow(img_rgb)
            ax.axis("off")

            found = False
            for box in preds.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                if conf < 0.4:
                    continue
                found = True
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="#00FF00", facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(x1, y1 - 6, f"{conf:.2f}",
                        color="white", fontsize=9, fontweight="bold",
                        bbox=dict(facecolor="#00AA00", alpha=0.7, pad=2))

            ax.set_title(
                f"{img_path.name[:25]}",
                fontsize=8,
                color="green" if found else "red"
            )

        for j in range(len(samples), 6):
            axes[j].axis("off")

        plt.tight_layout()
        sample_path = BASE_DIR / "sample_detections.png"
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        print(f"Sample detections saved to {sample_path}")
        plt.show()

    print("\nDone.")