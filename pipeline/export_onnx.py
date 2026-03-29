"""
export_onnx.py
──────────────
Exports both the detector and reader models from PyTorch (.pt) to ONNX (.onnx).

The exported files land next to the original weights:
  models/detector/weights/best.onnx
  models/reader/weights/best.onnx

"""


from pathlib import Path
from ultralytics import YOLO
 
ROOT        = Path(__file__).parent.parent
DETECTOR_PT = ROOT / "models" / "detector" / "weights" / "best.pt"
READER_PT   = ROOT / "models" / "reader"   / "weights" / "best.pt"
 
def export(pt_path: Path):
    print(f"\nExporting: {pt_path.parent.parent.name} model")
    model = YOLO(str(pt_path))
    model.export(
        format   = "onnx",
        imgsz    = 640,
        dynamic  = False,
        simplify = True,
        half = True,
        device = 0,
        opset    = 11,      # stable opset, wide runtime support
    )
    onnx_path = pt_path.with_suffix(".onnx")
    if onnx_path.exists():
        pt_mb   = pt_path.stat().st_size   / 1024 / 1024
        onnx_mb = onnx_path.stat().st_size / 1024 / 1024
        print(f"  PyTorch : {pt_mb:.1f} MB")
        print(f"  ONNX    : {onnx_mb:.1f} MB")
    else:
        print("  Export failed.")
 
if __name__ == "__main__":
    print("=" * 50)
    print("  ONNX Export")
    print("=" * 50)
    for pt_path in [DETECTOR_PT, READER_PT]:
        if not pt_path.exists():
            print(f"\nSkipping — not found: {pt_path}")
            continue
        export(pt_path)
    print("\nDone. Run benchmark.py next.")
 