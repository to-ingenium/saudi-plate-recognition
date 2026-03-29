# Saudi Plate Recognition — ANPR Pipeline

An end-to-end Automatic Number Plate Recognition (ANPR) system for Saudi license plates. The pipeline chains two YOLOv8s models: a plate detector that locates plates in full scenes, and a character reader that identifies individual Arabic letters and digits from the cropped plate region.

---

## Pipeline

```
Full scene image / video frame
        │
        ▼
 ┌─────────────┐
 │  Detector   │  YOLOv8s — finds and crops the plate region
 └─────────────┘
        │  plate crop (resized to 320×160)
        ▼
 ┌─────────────┐
 │   Reader    │  YOLOv8s — reads individual characters left to right
 └─────────────┘
        │
        ▼
 Annotated output (bounding box + plate text + confidence)
```

---

## Models

### Plate Detector
Trained on ~4,500 images of Saudi vehicles in real-world conditions — parking lots, streets, varying lighting and angles.

| Metric | Score |
|---|---|
| mAP50 | 97.0% |
| mAP50-95 | 82.0% |
| Precision | 95.0% |
| Recall | 95.0% |

### Character Reader
Trained on cropped Saudi plate images with individual character annotations covering Arabic letters and digits.

| Metric | Score |
|---|---|
| mAP50 | 97.1% |
| mAP50-95 | 74.0% |
| Precision | 94.7% |
| Recall | 93.0% |

---

## Benchmark — PyTorch vs ONNX

Both models are exported to ONNX for edge deployment. The benchmark compares GPU inference (PyTorch, development/server) against CPU inference (ONNX, edge devices with no GPU).

| Model | Format | Size (MB) | Avg ms/frame | Notes |
|---|---|---|---|---|
| Detector | PT | 21.5 | 7.0 | GPU · PyTorch stack · development/server |
| Detector | ONNX | 21.4 | 88.5 | CPU · edge runtime · no GPU required |
| Reader | PT | 21.5 | 7.8 | GPU · PyTorch stack · development/server |
| Reader | ONNX | 21.4 | 88.2 | CPU · edge runtime · no GPU required |

PT benchmarked on NVIDIA GeForce RTX 3060 Ti. ONNX benchmarked on CPU via `onnxruntime` to simulate edge deployment conditions.

---

## Challenges and Shortcomings

### Data quality
The detector training dataset (~4,500 images) contained a significant number of partially annotated plates — bounding boxes that covered only a portion of the plate region. These were partially cleaned but not fully resolved due to dataset size. This likely contributes to the gap between mAP50 (97%) and mAP50-95 (82%) — the model finds plates reliably but box tightness suffers on noisier samples.

### Reader accuracy on real-world input
The character reader was trained on clean, close-up cropped plates. In the full pipeline, the crop it receives is a resized region extracted from a compressed video frame — lower resolution and sometimes blurry. This degrades character recognition noticeably on low-quality footage. The reader performs well on high-quality images but struggles with:
- Heavily compressed video (WhatsApp, social media)
- Far-distance plates where the crop is small
- Motion blur from moving vehicles

### Arabic character ordering
Saudi plates contain both Arabic and Latin characters. The pipeline sorts detected characters left-to-right by bounding box position, which works for the Latin side but can produce incorrect ordering for the Arabic side which reads right-to-left. A proper implementation would require split handling for each half of the plate.

---

## Project Structure

```
saudi-plate-recognition/
├── models/
│   ├── detector/
│   │   ├── data/          — training dataset (not tracked in git)
│   │   ├── weights/       — best.pt, best.onnx (not tracked in git)
│   │   ├── runs/          — results.csv
│   │   └── scripts/       — prepare_data.py, train.py
│   └── reader/
│       ├── data/
│       ├── weights/
│       ├── runs/
│       └── scripts/
├── pipeline/
│   ├── detection.py       — main inference script (images + video)
│   ├── export_onnx.py     — exports both models to ONNX
│   └── benchmark.py       — PT vs ONNX speed and size comparison
├── assets/
│   ├── test_samples/      — sample inputs and outputs
│   ├── benchmark_results.txt
│   ├── sample_detections.png
│   └── training_results.png
├── requirements.txt
└── .gitignore
```


---

## Dataset Sources

- Detector: [Saudi Licence Plates — Roboflow Universe](https://universe.roboflow.com/cars-o2wwf/saudi-licence-plates/dataset/3) (CC BY 4.0)
- Character reader: Saudi plate Characters [Kaggle](https://www.kaggle.com/datasets/riotulab/saudi-license-plate-characters/data)
