"""
benchmark.py
────────────
Compares PyTorch (.pt) vs ONNX (.onnx) inference for both models.

How it works:
  - PT models run via Ultralytics + PyTorch (GPU)
  - ONNX models run via onnxruntime directly (CPU) — the actual edge runtime
  - 5 warmup runs to avoid cold-start bias, then 50 timed runs
  - Reports size (MB) and avg inference time (ms/frame)

This reflects the real-world tradeoff:
  PT  = full PyTorch stack, GPU, heavy
  ONNX = lightweight runtime, portable, edge-ready

"""

import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import onnxruntime as ort

ROOT       = Path(__file__).parent.parent
MODELS     = {
    "Detector": {
        "pt":   ROOT / "models" / "detector" / "weights" / "best.pt",
        "onnx": ROOT / "models" / "detector" / "weights" / "best.onnx",
    },
    "Reader": {
        "pt":   ROOT / "models" / "reader" / "weights" / "best.pt",
        "onnx": ROOT / "models" / "reader" / "weights" / "best.onnx",
    },
}
TEST_IMAGE_DIR = ROOT / "assets" / "test_samples"
OUTPUT_FILE    = ROOT / "assets" / "benchmark_results.txt"

WARMUP_RUNS = 5
TIMED_RUNS  = 50

# ── Load test image ───────────────────────────────────────────────────────────
image_files = list(TEST_IMAGE_DIR.glob("*.jpg")) + list(TEST_IMAGE_DIR.glob("*.png"))
if not image_files:
    raise FileNotFoundError(f"No images found in {TEST_IMAGE_DIR}")

test_image_path = image_files[0]
img_bgr = cv2.imread(str(test_image_path))

# Preprocessed image for ONNX (CHW, normalized, batch dim)
img_resized = cv2.resize(img_bgr, (640, 640))
img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_onnx    = (img_rgb.astype(np.float16) / 255.0)
img_onnx    = np.transpose(img_onnx, (2, 0, 1))[np.newaxis]  # (1, 3, 640, 640)

print(f"Test image : {test_image_path.name}")
print(f"Warmup     : {WARMUP_RUNS} runs")
print(f"Timed      : {TIMED_RUNS} runs\n")

# ── PT benchmark via Ultralytics ──────────────────────────────────────────────
def benchmark_pt(model_path: Path) -> float:
    model = YOLO(str(model_path))
    for _ in range(WARMUP_RUNS):
        model(img_bgr, verbose=False)
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        model(img_bgr, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

# ── ONNX benchmark via onnxruntime ────────────────────────────────────────────
def benchmark_onnx(model_path: Path) -> float:
    sess = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"]  # edge runtime = CPU
    )
    input_name = sess.get_inputs()[0].name
    for _ in range(WARMUP_RUNS):
        sess.run(None, {input_name: img_onnx})
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: img_onnx})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times))

# ── Run all benchmarks ────────────────────────────────────────────────────────
print("=" * 62)
print("  Benchmarking")
print("=" * 62)

results = []
for model_name, paths in MODELS.items():
    print(f"\n{model_name}:")
    for fmt, path in paths.items():
        if not path.exists():
            print(f"  {fmt.upper()} — not found, skipping")
            continue
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"  {fmt.upper()} ({size_mb:.1f} MB) — running...", end=" ", flush=True)
        avg_ms = benchmark_pt(path) if fmt == "pt" else benchmark_onnx(path)
        print(f"{avg_ms:.1f} ms")
        results.append({"name": model_name, "format": fmt,
                         "size_mb": size_mb, "avg_ms": avg_ms})

# ── Build table ───────────────────────────────────────────────────────────────
header  = f"\n{'Model':<12} {'Format':<8} {'Size (MB)':<12} {'Avg ms/frame':<16} Notes"
divider = "─" * 65
rows    = []

for model_name in MODELS:
    group  = [r for r in results if r["name"] == model_name]
    pt_r   = next((r for r in group if r["format"] == "pt"),   None)
    onnx_r = next((r for r in group if r["format"] == "onnx"), None)

    for r in group:
        if r["format"] == "pt":
            note = "GPU · PyTorch stack · development/server"
        else:
            note = f"CPU · edge runtime · no GPU required · {(1 - r['size_mb']/pt_r['size_mb'])*100:+.0f}% size" if pt_r else "CPU · edge runtime"
        rows.append(
            f"{r['name']:<12} {r['format'].upper():<8} "
            f"{r['size_mb']:<12.1f} {r['avg_ms']:<16.1f} {note}"
        )

table = "\n".join([header, divider] + rows + [divider])
print(table)

# ── Save ──────────────────────────────────────────────────────────────────────
OUTPUT_FILE.parent.mkdir(exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    f.write("Saudi Plate Recognition — Benchmark Results\n")
    f.write(f"Test image : {test_image_path.name}\n")
    f.write(f"Warmup: {WARMUP_RUNS} | Timed: {TIMED_RUNS}\n")
    f.write(table + "\n")

print(f"\nSaved → {OUTPUT_FILE}")