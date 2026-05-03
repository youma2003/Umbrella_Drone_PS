import json, os, random, shutil
from pathlib import Path

DATASET = Path(r"C:\Users\Maryem\Desktop\ps-drone\dataset\images\YOLODataset")
SRC_IMG  = DATASET / "images" / "train"
DST_LBL_TRAIN = DATASET / "labels" / "train"
DST_LBL_VAL   = DATASET / "labels" / "val"
DST_IMG_VAL   = DATASET / "images" / "val"

DST_LBL_TRAIN.mkdir(parents=True, exist_ok=True)
DST_LBL_VAL.mkdir(parents=True, exist_ok=True)
DST_IMG_VAL.mkdir(parents=True, exist_ok=True)

json_files = list(SRC_IMG.glob("*.json"))
print(f"JSONs trouvés : {len(json_files)}")

converted, skipped = 0, 0

for jf in json_files:
    with open(jf, encoding="utf-8") as f:
        data = json.load(f)

    W = data.get("imageWidth")
    H = data.get("imageHeight")
    shapes = data.get("shapes", [])

    if not shapes or not W or not H:
        skipped += 1
        continue

    lines = []
    for shape in shapes:
        pts = shape["points"]  # [[x1,y1],[x2,y2]]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        cx = ((x1 + x2) / 2) / W
        cy = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W
        h  = (y2 - y1) / H

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))

        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    txt_path = DST_LBL_TRAIN / (jf.stem + ".txt")
    txt_path.write_text("\n".join(lines))
    converted += 1

print(f"Convertis : {converted} | Ignorés : {skipped}")

# --- Move 20% to val ---
all_txts = list(DST_LBL_TRAIN.glob("*.txt"))
random.seed(42)
random.shuffle(all_txts)
val_count = max(1, int(len(all_txts) * 0.2))
val_txts  = all_txts[:val_count]

moved = 0
for lbl in val_txts:
    # Move label
    shutil.move(str(lbl), str(DST_LBL_VAL / lbl.name))

    # Move matching image (try jpg then png)
    for ext in (".jpg", ".png"):
        img_src = SRC_IMG / (lbl.stem + ext)
        if img_src.exists():
            shutil.move(str(img_src), str(DST_IMG_VAL / img_src.name))
            moved += 1
            break

print(f"Val : {val_count} labels déplacés, {moved} images déplacées")
print(f"Train final : {len(all_txts) - val_count} | Val final : {val_count}")
