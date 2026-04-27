import os, shutil, random
from pathlib import Path
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  
DATA_DIR = BASE_DIR / "data"

DATASETS = [
    {
        "workspace": "roboflow-universe-projects",
        "project":   "license-plate-recognition-rxg4e",
        "version":   13,
    },
]

# ── Download ──────────────────────────────────────────────────────────────────
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])

downloaded = []
for ds in DATASETS:
    project = rf.workspace(ds["workspace"]).project(ds["project"])
    data    = project.version(ds["version"]).download(
                "yolov8",
                location=str(DATA_DIR)
              )
    downloaded.append((data.location, ds.get("class_map", {})))

# ── Collect + remap ───────────────────────────────────────────────────────────
all_pairs = []

for ds_path, class_map in downloaded:
    for split in ["train", "valid", "test"]:
        img_dir = Path(ds_path) / split / "images"
        lbl_dir = Path(ds_path) / split / "labels"
        if not img_dir.exists():
            continue
        for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            lines = lbl.read_text().strip().splitlines()
            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    new_lines.append("0 " + " ".join(parts[1:]))
            if new_lines:
                all_pairs.append((img, "\n".join(new_lines)))

print(f"Total usable images: {len(all_pairs)}")

# ── Shuffle + split 70/20/10 ──────────────────────────────────────────────────
random.seed(42)
random.shuffle(all_pairs)

n       = len(all_pairs)
n_train = int(n * 0.70)
n_val   = int(n * 0.20)

splits = {
    "train": all_pairs[:n_train],
    "valid": all_pairs[n_train:n_train + n_val],
    "test":  all_pairs[n_train + n_val:]
}

# ── Write directly into data/ ─────────────────────────────────────────────────
for split, pairs in splits.items():
    img_out = DATA_DIR / split / "images"
    lbl_out = DATA_DIR / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img_path, label_text in pairs:
        shutil.copy(img_path, img_out / img_path.name)
        (lbl_out / (img_path.stem + ".txt")).write_text(label_text)

    print(f"{split}: {len(pairs)} images")

# ── data.yaml ─────────────────────────────────────────────────────────────────
(DATA_DIR / "data.yaml").write_text(f"""train: {DATA_DIR / 'train' / 'images'}
val:   {DATA_DIR / 'valid' / 'images'}
test:  {DATA_DIR / 'test'  / 'images'}

nc: 1
names: ['license_plate']
""")

# ── Clean up raw download ─────────────────────────────────────────────────────
shutil.rmtree(DATA_DIR)
print(f"Done — data ready at: {DATA_DIR}")


