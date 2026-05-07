"""
convert_and_add.py
Convertit les annotations LabelMe (JSON) en format YOLO (.txt)
et déplace les images + labels dans le YOLODataset (train/val).

Usage :
    python convert_and_add.py <dossier_images_labelisées>

Exemple :
    python convert_and_add.py images/live_captures
    python convert_and_add.py images/altitude_captures

Le script :
  1. Lit chaque .json LabelMe dans le dossier
  2. Convertit les rectangles en format YOLO normalisé
  3. Copie 80% en train, 20% en val
"""

import json, os, shutil, random, sys
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = Path(__file__).parent / "images" / "YOLODataset"
TRAIN_IMG    = DATASET_ROOT / "images" / "train"
VAL_IMG      = DATASET_ROOT / "images" / "val"
TRAIN_LBL    = DATASET_ROOT / "labels" / "train"
VAL_LBL      = DATASET_ROOT / "labels" / "val"
VAL_RATIO    = 0.20   # 20% des images vont en val
CLASS_MAP    = {"person": 0}   # ajoute d'autres classes ici si besoin

# ── ARGUMENT ──────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python convert_and_add.py <dossier>")
    print("Ex:    python convert_and_add.py images/live_captures")
    sys.exit(1)

src = Path(__file__).parent / sys.argv[1]
if not src.exists():
    print(f"[ERREUR] Dossier introuvable : {src}")
    sys.exit(1)

jsons = sorted(src.glob("*.json"))
if not jsons:
    print(f"[ERREUR] Aucun fichier .json trouvé dans {src}")
    print("         Vérifie que tu as sauvegardé les annotations dans LabelMe.")
    sys.exit(1)

print(f"[INFO] {len(jsons)} fichiers JSON trouvés dans {src}")

# ── CONVERSION ────────────────────────────────────────────────────────────────
converted = []   # (img_path, txt_content)
skipped   = 0

for json_path in jsons:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    img_w = data["imageWidth"]
    img_h = data["imageHeight"]

    lines = []
    for shape in data["shapes"]:
        label = shape["label"].strip().lower()
        if label not in CLASS_MAP:
            print(f"  [SKIP] label inconnu '{label}' dans {json_path.name}")
            continue

        cls_id = CLASS_MAP[label]

        if shape["shape_type"] == "rectangle":
            pts = shape["points"]   # [[x1,y1],[x2,y2]]
            x1 = min(pts[0][0], pts[1][0])
            y1 = min(pts[0][1], pts[1][1])
            x2 = max(pts[0][0], pts[1][0])
            y2 = max(pts[0][1], pts[1][1])
        elif shape["shape_type"] == "polygon":
            xs = [p[0] for p in shape["points"]]
            ys = [p[1] for p in shape["points"]]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
        else:
            print(f"  [SKIP] shape_type '{shape['shape_type']}' ignoré")
            continue

        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        # Clamp au cas où le rectangle dépasse les bords
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        skipped += 1
        continue

    # Cherche l'image correspondante (.jpg ou .png)
    img_path = None
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = json_path.with_suffix(ext)
        if candidate.exists():
            img_path = candidate
            break

    if img_path is None:
        print(f"  [SKIP] image introuvable pour {json_path.name}")
        skipped += 1
        continue

    converted.append((img_path, "\n".join(lines)))

print(f"[INFO] {len(converted)} images converties, {skipped} ignorées")

# ── RÉPARTITION TRAIN / VAL ───────────────────────────────────────────────────
random.shuffle(converted)
split = max(1, int(len(converted) * VAL_RATIO))
val_set   = converted[:split]
train_set = converted[split:]

for folder in (TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL):
    folder.mkdir(parents=True, exist_ok=True)

def copy_pair(img_path, txt_content, img_dir, lbl_dir):
    dest_img = img_dir / img_path.name
    dest_lbl = lbl_dir / (img_path.stem + ".txt")
    shutil.copy2(img_path, dest_img)
    dest_lbl.write_text(txt_content, encoding="utf-8")

for img_path, txt in train_set:
    copy_pair(img_path, txt, TRAIN_IMG, TRAIN_LBL)

for img_path, txt in val_set:
    copy_pair(img_path, txt, VAL_IMG, VAL_LBL)

print(f"\n[DONE] Ajouté au dataset :")
print(f"  train : {len(train_set)} images  -> {TRAIN_IMG}")
print(f"  val   : {len(val_set)} images   -> {VAL_IMG}")
print(f"\nRelance l'entraînement avec le même dataset.yaml.")
