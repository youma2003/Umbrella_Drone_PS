# Drone Autonome de Suivi de Piéton
## Webots · YOLOv8 · Contrôleur PI · Vision par ordinateur

> **Projet** : Simulation complète d'un drone Mavic 2 Pro capable de détecter un piéton depuis les airs (vue de dessus), de se stabiliser exactement au-dessus de sa tête, puis de le suivre de façon autonome quelle que soit sa vitesse de marche.

---

## Table des matières

1. [Résumé du projet](#1-résumé-du-projet)
2. [Environnement & matériel](#2-environnement--matériel)
3. [Technologies utilisées](#3-technologies-utilisées)
4. [Structure du projet](#4-structure-du-projet)
5. [Architecture système complète](#5-architecture-système-complète)
6. [Simulation Webots](#6-simulation-webots)
7. [Dataset — collecte et statistiques](#7-dataset--collecte-et-statistiques)
8. [Labelling — processus complet](#8-labelling--processus-complet)
9. [Modèle YOLO — entraînement](#9-modèle-yolo--entraînement)
10. [Formules géométriques](#10-formules-géométriques)
11. [Contrôleur PI — théorie et implémentation](#11-contrôleur-pi--théorie-et-implémentation)
12. [Machine d'états](#12-machine-détats)
13. [Dead reckoning](#13-dead-reckoning)
14. [Fenêtre de debug OpenCV](#14-fenêtre-de-debug-opencv)
15. [Comparaison des approches de contrôle testées](#15-comparaison-des-approches-de-contrôle-testées)
16. [Analyse de performance](#16-analyse-de-performance)
17. [Problèmes rencontrés et solutions](#17-problèmes-rencontrés-et-solutions)
18. [Paramètres finaux](#18-paramètres-finaux)
19. [Limitations et travaux futurs](#19-limitations-et-travaux-futurs)
20. [Glossaire](#20-glossaire)
21. [Installation et lancement](#21-installation-et-lancement)

---

## 1. Résumé du projet

Ce projet implémente un système de **suivi autonome de piéton par drone** dans un environnement de simulation 3D. Le drone est un **DJI Mavic 2 Pro** simulé dans **Webots**. Il utilise sa caméra embarquée orientée vers le bas pour détecter la tête d'un piéton grâce à un modèle **YOLOv8** entraîné sur un dataset personnalisé (vue top-down), puis applique un **contrôleur PI** pour maintenir le drone exactement au-dessus du piéton.

### Objectifs atteints

| Objectif | Statut |
|----------|--------|
| Décollage automatique et stabilisation à 4m | ✅ |
| Détection de la tête du piéton (YOLOv8 custom) | ✅ |
| Centrage du drone au-dessus de la tête | ✅ |
| Suivi du piéton à marche lente (0.25 m/s) | ✅ |
| Suivi du piéton à marche rapide (0.875 m/s) | ✅ |
| Stabilisation automatique quand piéton immobile | ✅ |
| Reprise du suivi après perte de cible | ✅ |
| Fenêtre de visualisation temps réel OpenCV | ✅ |
| Contrôle manuel clavier conservé | ✅ |

### Résultat observé dans les logs

```
[STATE] Piéton détecté → FOLLOW
[FOLLOW] dist=1.85m I=(+0.00,+0.00) | cmd=(-0.12,+0.68)
[FOLLOW] dist=1.30m I=(+0.16,+1.08) | cmd=(-0.19,+1.36)
[FOLLOW] dist=0.39m I=(+0.21,+1.48) | cmd=(-0.09,+0.84)
[STABLE] dist=0.04m I=(+0.07,+0.54) | alt=4.45m       ← drone exactement au-dessus
[FOLLOW] dist=0.36m I=(+0.03,+0.21) | cmd=(+0.03,-0.09) ← piéton repart → drone suit
```

---

## 2. Environnement & matériel

| Composant | Détail |
|-----------|--------|
| **OS** | Windows 11 Pro |
| **CPU** | Intel Core i7-13650HX (13e génération) |
| **RAM** | (CPU uniquement, pas de GPU utilisé) |
| **Python** | 3.13 |
| **Webots** | R2023b |
| **Simulateur physique** | ODE (Open Dynamics Engine) intégré à Webots |

> Tout l'entraînement YOLO et l'inférence temps réel tournent sur **CPU uniquement**. La durée d'inférence YOLO mesurée : **~40ms/frame**.

---

## 3. Technologies utilisées

| Outil | Version | Usage dans le projet |
|-------|---------|----------------------|
| **Python** | 3.13 | Langage principal des deux contrôleurs |
| **Webots** | R2023b | Moteur de simulation 3D (drone, piéton, physique) |
| **Ultralytics YOLOv8n** | 8.4.41 | Détection de tête vue de dessus |
| **PyTorch** | 2.11.0+cpu | Backend d'inférence et d'entraînement YOLO |
| **OpenCV (cv2)** | — | Conversion image, fenêtre debug temps réel |
| **NumPy** | — | Reshape buffer caméra BGRA→RGB |
| **LabelMe** | 6.1.3 | Annotation initiale (JSON, convertie ensuite) |
| **LabelImg** | — | Annotation YOLO native (format .txt direct) |

---

## 4. Structure du projet

```
ps-drone/
│
├── controllers/
│   ├── mavic2pro_controller/
│   │   ├── mavic2pro_controller.py   ← contrôleur principal (760 lignes)
│   │   ├── best.pt                   ← modèle YOLO entraîné (6.2 MB)
│   │   └── yolov8n.pt                ← base model YOLOv8 Nano
│   │
│   └── pedestrian_controller/
│       └── pedestrian_controller.py  ← déplacement piéton via Supervisor API
│
├── dataset/
│   ├── convert_labels.py             ← script conversion LabelMe JSON → YOLO
│   └── images/YOLODataset/
│       ├── dataset.yaml              ← configuration entraînement YOLO
│       ├── images/
│       │   ├── train/  (457 images .jpg)
│       │   └── val/    (118 images .jpg)
│       └── labels/
│           ├── train/  (283 labels .txt)
│           └── val/    (70 labels .txt)
│
├── worlds/
│   └── mavic_2_pro.wbt               ← scène Webots (drone + piéton + sol)
│
├── protos/                           ← modèles 3D Webots
└── README.md
```

---

## 5. Architecture système complète

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SIMULATION WEBOTS                            │
│                                                                     │
│  ┌──────────────────┐          ┌────────────────────────────────┐  │
│  │ pedestrian_ctrl  │          │     mavic2pro_controller       │  │
│  │  (Supervisor)    │          │                                │  │
│  │                  │  scène   │  ┌─────────┐  ┌───────────┐  │  │
│  │ pos += 0.007/step│ ──────→  │  │ Caméra  │  │  Capteurs │  │  │
│  │ setSFVec3f(pos)  │  3D      │  │ 640×480 │  │ IMU/GPS   │  │  │
│  └──────────────────┘          │  └────┬────┘  └─────┬─────┘  │  │
│                                │       │              │         │  │
│                                │  ┌────▼────────────▼──────┐  │  │
│                                │  │      YOLO (toutes les   │  │  │
│                                │  │      2 frames = 16ms)   │  │  │
│                                │  │   BGRA→RGB → inférence  │  │  │
│                                │  │   → bounding boxes      │  │  │
│                                │  └────────────┬────────────┘  │  │
│                                │               │               │  │
│                                │  ┌────────────▼────────────┐  │  │
│                                │  │  GÉOMÉTRIE CAMÉRA       │  │  │
│                                │  │  off_x,off_y → dx_m,dy_m│  │  │
│                                │  │  dist = √(dx²+dy²)      │  │  │
│                                │  └────────────┬────────────┘  │  │
│                                │               │               │  │
│                                │  ┌────────────▼────────────┐  │  │
│                                │  │  MACHINE D'ÉTATS        │  │  │
│                                │  │  TAKEOFF/SEARCH/FOLLOW  │  │  │
│                                │  └────────────┬────────────┘  │  │
│                                │               │               │  │
│                                │  ┌────────────▼────────────┐  │  │
│                                │  │  CONTRÔLEUR PI          │  │  │
│                                │  │  integral += dx×dt_s    │  │  │
│                                │  │  cmd = KP×err + KI×∫err │  │  │
│                                │  └────────────┬────────────┘  │  │
│                                │               │               │  │
│                                │  ┌────────────▼────────────┐  │  │
│                                │  │  PID MOTEURS (code C)   │  │  │
│                                │  │  FL / FR / RL / RR      │  │  │
│                                │  └─────────────────────────┘  │  │
│                                └────────────────────────────────┘  │
│                                                                     │
│                    ┌──────────────────────────┐                    │
│                    │  DEBUG OpenCV (temps réel)│                    │
│                    │  bounding box + vecteur   │                    │
│                    │  erreur + état + distance │                    │
│                    └──────────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Simulation Webots

### Drone : Mavic 2 Pro

Le modèle Webots du Mavic 2 Pro est un modèle officiel fourni par Cyberbotics. Il embarque :

| Capteur | Données fournies | Fréquence |
|---------|-----------------|-----------|
| `inertial unit` | roll, pitch, yaw | chaque step (8ms) |
| `gps` | x, y, altitude | chaque step (8ms) |
| `gyro` | vitesses angulaires roll_v, pitch_v | chaque step (8ms) |
| `camera` | image BGRA 640×480 | chaque step (8ms) |
| `keyboard` | touches pressées | chaque step (8ms) |

**Moteurs :** 4 hélices à vitesse variable + 2 servos caméra (roll/pitch gimbal).

**Stabilisation caméra** à chaque frame :
```python
cam_roll_motor.setPosition(-0.115 * roll_v)
cam_pitch_motor.setPosition(-0.1   * pitch_v)
```

### Configuration caméra (depuis mavic_2_pro.wbt)

```
Camera {
  translation    0 0 -0.05
  rotation       0 1 0 1.5708    ← rotation de π/2 → pointe vers le bas
  fieldOfView    1               ← FOV horizontal = 1.0 radian (57.3°)
  width          640
  height         480
}
```

### PID de stabilisation vol (constantes code C original conservées)

```python
K_VERTICAL_THRUST = 68.5   # poussée de base pour maintenir altitude
K_VERTICAL_OFFSET = 0.6    # offset statique
K_VERTICAL_P      = 3.0    # gain proportionnel altitude
K_ROLL_P          = 50.0   # gain stabilisation roulis
K_PITCH_P         = 30.0   # gain stabilisation tangage
```

Ces constantes sont identiques à celles du contrôleur C officiel Webots. Le contrôle de suivi s'ajoute sous forme de **perturbations** (`roll_disturbance`, `pitch_disturbance`) injectées dans ce PID existant, ce qui garantit la stabilité en vol quel que soit l'état du suivi.

### Contrôleur piéton (Supervisor API)

```python
from controller import Supervisor

robot = Supervisor()
translation_field = robot.getSelf().getField("translation")

while robot.step(timestep) != -1:
    pos = robot.getSelf().getPosition()
    translation_field.setSFVec3f([pos[0] + 0.007, pos[1], pos[2]])
```

Le piéton est déplacé par **téléportation directe** (sans physique). Cela a des conséquences importantes sur le contrôleur du drone (voir section 11).

| Paramètre `step` | Vitesse réelle (8ms/step) | Comportement du suivi |
|-----------------|--------------------------|----------------------|
| `0.002` | **0.25 m/s** (marche lente) | Suivi parfait, STABLE rapide |
| `0.005` | **0.625 m/s** (marche normale) | Suivi fiable |
| `0.007` | **0.875 m/s** (marche rapide) | Suivi stable avec PI |
| `0.010` | **1.25 m/s** (limite) | Suivi tendu, occasionnellement perdu |

---

## 7. Dataset — collecte et statistiques

### Méthode de collecte

Les images ont été capturées **manuellement** sous forme de captures d'écran de la fenêtre caméra de Webots durant la simulation. La caméra est en vue top-down (légèrement inclinée vers l'avant), ce qui donne une vue typique :

```
  ┌─────────────────────────────┐
  │                             │
  │      sol / texture          │
  │                             │
  │    ┌──────┐                 │
  │    │ tête │  ← petit ovale  │
  │    └──────┘                 │
  │   /        \                │
  │  ( épaules  )               │
  │                             │
  └─────────────────────────────┘
        Vue caméra drone
```

### Statistiques du dataset final

| Métrique | Valeur |
|----------|--------|
| **Total images annotées** | 597 |
| Images train | 457 (après split) |
| Images val | 118 (après split) |
| Labels train utilisés | 283 |
| Labels val utilisés | 70 |
| Images sans annotation (backgrounds) | ~244 |
| Format images | PNG → converti JPG |
| Résolution images | ~550×415 px (variable) |
| Format labels | YOLO `.txt` (cx cy w h normalisés) |
| Classe unique | `person` (classe 0) |

> **Note** : 244 images sans annotation (backgrounds) dans le dataset sont conservées intentionnellement — elles aident le modèle à apprendre à **ne pas détecter** ce qui n'est pas un piéton (réduction des faux positifs).

### Distribution des tailles de boîtes (après correction)

```
Taille boîte (% de l'image) :
  < 10%  : ████████████████ 38%   ← tête seule, parfait
  10-20% : ████████████     29%   ← tête + léger contexte
  20-30% : ████████         19%   ← tête visible + quelques pixels épaules
  > 30%  : ████             14%   ← boîtes imparfaites (à corriger si retrain)

Idéal cible : 100% < 20%
```

---

## 8. Labelling — processus complet

### Pourquoi labelliser seulement la tête

La caméra pointant vers le bas depuis 4 mètres, seules la **tête** et **les épaules** sont visibles.

**Problème initial** : le premier dataset avait des boîtes couvrant la tête **ET** les épaules (60-77% de l'image). Le modèle YOLO apprenait à placer le centre de sa prédiction au centre de ces boîtes → le centre tombait **sur les épaules**, pas sur la tête. Le drone suivait donc les épaules, pas la tête.

```
ANCIEN LABELLING — boîte tête+épaules :    CORRECT — boîte tête uniquement :

  ┌──────────────────┐                       ┌──────┐
  │   ████ (tête)    │                       │ ████ │  cy = tête ✓
  │                  │  cy = épaules ✗       └──────┘
  │ ██████ (épaules) │  boîte : 64×77%        boîte : ~10×15%
  └──────────────────┘
```

### Outil utilisé : LabelImg

```bash
pip install labelImg
labelImg
```

**Workflow correct (ordre des étapes OBLIGATOIRE) :**

```
1. Lancer labelImg
2. Cliquer sur le bouton format → sélectionner "YOLO"
3. Change Save Dir → dossier labels/train/
4. Open Dir        → dossier images/train/
5. View → Auto Save Mode ✓
6. Créer classes.txt dans le dossier images/ :
     person
7. Pour chaque image :
   - Touche W → dessiner boîte autour de la tête
   - Boîte = presque carrée (tête vue de dessus ≈ cercle)
   - Ne pas inclure les épaules
   - Touche D → image suivante (sauvegarde automatique)
```

### Problème rencontré : format JSON au lieu de YOLO

Le premier labelling a utilisé **LabelMe** (pas LabelImg). LabelMe sauvegarde en JSON :

```json
{
  "version": "6.1.3",
  "shapes": [{
    "label": "personne",
    "points": [[155.6, 72.7], [0.0, 357.1]],
    "shape_type": "rectangle"
  }],
  "imageWidth": 546,
  "imageHeight": 414
}
```

Ce format est incompatible avec YOLO directement.

### Script de conversion LabelMe JSON → YOLO (`convert_labels.py`)

```python
import json, os, shutil
from pathlib import Path

for jf in Path("images/train").glob("*.json"):
    with open(jf) as f:
        data = json.load(f)

    W, H = data["imageWidth"], data["imageHeight"]

    for shape in data["shapes"]:
        pts = shape["points"]
        x1, x2 = min(p[0] for p in pts), max(p[0] for p in pts)
        y1, y2 = min(p[1] for p in pts), max(p[1] for p in pts)

        cx = ((x1 + x2) / 2) / W     # centre normalisé [0,1]
        cy = ((y1 + y2) / 2) / H
        w  = (x2 - x1) / W           # largeur normalisée [0,1]
        h  = (y2 - y1) / H           # hauteur normalisée [0,1]

        # Format YOLO : class_id cx cy w h
        line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    Path("labels/train").mkdir(exist_ok=True)
    (Path("labels/train") / (jf.stem + ".txt")).write_text(line)

# Split automatique 80% train / 20% val
```

**Split 80/20 automatique** : après conversion, le script déplace aléatoirement 20% des paires image+label vers `val/`.

---

## 9. Modèle YOLO — entraînement

### Architecture : YOLOv8 Nano

YOLOv8n est le plus petit modèle de la famille YOLOv8, avec 3M paramètres. Il est choisi pour :
- Inférence rapide sur CPU (~40ms/frame)
- Taille légère (6.2 MB après fine-tuning)
- Performances suffisantes pour une seule classe simple (tête vue de dessus)

```
Architecture YOLOv8n (résumé) :
  Backbone  : CSPDarknet (Conv + C2f blocks)
  Neck      : FPN + PAN (feature pyramid)
  Head      : Detect (3 échelles : 80×80, 40×40, 20×20)
  Paramètres: 3,011,043
  GFLOPs    : 8.2
```

### Configuration d'entraînement

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # partir des poids pré-entraînés COCO
model.train(
    data    = 'dataset/images/YOLODataset/dataset.yaml',
    epochs  = 50,
    imgsz   = 640,
    batch   = 8,        # réduit de 16 → 8 (crash RAM CPU à 16)
    device  = 'cpu',
    workers = 0,        # nécessaire sur Windows
    cache   = False
)
```

**dataset.yaml :**
```yaml
path: C:/Users/Maryem/Desktop/ps-drone/dataset/images/YOLODataset
train: images/train
val:   images/val
nc: 1
names: ['person']
```

### Évolution de l'entraînement

| Phase | Epochs | box_loss | cls_loss | mAP50 |
|-------|--------|----------|----------|-------|
| Début | 1 | ~1.0 | ~4.3 | — |
| Mi-parcours | 24 | ~0.70 | ~0.47 | 0.410 |
| **Final** | **50** | **~0.68** | **~0.45** | **0.455** |

L'entraînement a crashé à l'epoch 24 (exit code 5, dépassement mémoire). Reprise :

```python
model = YOLO('runs/detect/train-2/weights/last.pt')
model.train(resume=True)   # reprend depuis l'epoch 24
```

### Résultats finaux

| Métrique | Valeur | Interprétation |
|----------|--------|----------------|
| **mAP50** | **0.455** | Bonne détection à IoU=0.5 |
| **Recall** | **0.984** | Détecte 98.4% des piétons présents |
| Precision | 0.408 | Quelques faux positifs acceptables |
| mAP50-95 | 0.391 | Bonne localisation multi-seuils |
| Inférence | ~40ms | Viable pour temps réel sur CPU |

> Le **Recall élevé (0.984)** est prioritaire pour ce projet : mieux vaut une fausse détection occasionnelle qu'un piéton manqué. Le drone perdrait le suivi si le Recall était bas.

---

## 10. Formules géométriques

### Du pixel au monde réel

La caméra pointe vers le bas depuis une altitude `h`. Chaque pixel correspond à une distance réelle au sol. La conversion s'effectue via la trigonométrie du champ de vue :

```
Offset normalisé (image) :
  off_x = (cx_pixel - W/2) / (W/2)     ∈ [-1, 1]
  off_y = (cy_pixel - H/2) / (H/2)     ∈ [-1, 1]

Distance réelle au sol (mètres) :
  dx_m = off_x × altitude × tan(FOV_H / 2)
  dy_m = off_y × altitude × tan(FOV_V / 2)

  FOV_H = 1.000 rad  (lu depuis .wbt)
  FOV_V = 0.775 rad  (calculé : 2·atan(tan(0.5)·480/640))

Distance horizontale drone ↔ piéton :
  dist = √(dx_m² + dy_m²)
```

### Avantage de cette approche

Travailler en **mètres réels** (et non en pixels) permet :
- Un seuil de stabilité physique (`STABLE_DIST = 0.35m`) indépendant de l'altitude
- Des gains de contrôle (`KP`, `KI`) avec une signification physique (m → commande)
- Une adaptation automatique à l'altitude : monter le drone ne change pas les gains

### Champ visuel en fonction de l'altitude

| Altitude | Demi-largeur visible | Temps avant sortie frame (piéton 1 m/s) |
|----------|---------------------|------------------------------------------|
| 3m | 1.64m | 1.64s |
| **4m** | **2.18m** | **2.18s** ← altitude choisie |
| 5m | 2.73m | 2.73s |
| 6m | 3.28m | 3.28s |

L'altitude a été augmentée de 3m → 4m pour donner plus de marge de réaction quand le piéton marche vite.

---

## 11. Contrôleur PI — théorie et implémentation

### Pourquoi pas de feedforward vitesse ?

Le piéton est déplacé par **téléportation** (`setSFVec3f`) et non par physique. Cela crée une contrainte fondamentale :

```
Déplacement réel entre 2 détections YOLO (2 × 8ms) :
  Δx = 0.007 × 2 = 0.014m

Bruit de détection YOLO (jitter ≈ 20px sur 640px à 4m altitude) :
  σ_noise ≈ (20/640) × 2 × 4 × tan(0.5) = ±0.025m

Rapport signal/bruit = 0.014 / 0.025 = 0.56 < 1

→ La vitesse estimée (Δx/Δt) est dominée par le bruit de détection.
→ Un gain KFF × vx_brut amplifierait le bruit, pas le signal.
→ Décision : supprimer le feedforward, utiliser PI pur.
```

### Pourquoi l'intégrale à chaque frame (et non chaque YOLO)

YOLO tourne toutes les 2 frames (16ms). Si l'intégrale s'accumule seulement aux frames YOLO, elle croît 2× plus lentement qu'à chaque frame (8ms) :

```
Comparaison à dist=0.5m constant :

Intégrale YOLO (16ms) :      Intégrale frame (8ms) :
  t=1s : I = 0.5×1×0.35 = 0.175      t=1s : I = 0.5×1×0.35 = 0.175
  → faux, seulement 62 YOLO frames   → vrai, 125 frames × 0.008s
  
En pratique avec YOLO_EVERY=2 :
  I_yolo = dist × n_yolo × dt_yolo = dist × n × 0.016
  I_frame = dist × n_total × dt_s  = dist × 2n × 0.008

  → même résultat théorique, mais I_frame est mis à jour plus souvent
  → les transitions d'état (STABLE→FOLLOW) sont gérées plus finement
```

### Formule de commande complète

```
Chaque frame (8ms) :
  integral_x += dx_m × dt_s
  integral_y += dy_m × dt_s
  integral_x  = clamp(integral_x, -5.0, +5.0)    ← anti-windup
  integral_y  = clamp(integral_y, -5.0, +5.0)

Chaque frame YOLO (16ms) :
  pid_roll  = clamp( -dx_m × KP  -  integral_x × KI , -3.0, +3.0 )
  pid_pitch = clamp(  dy_m × KP  +  integral_y × KI , -3.0, +3.0 )

  smooth_roll_p  += SMOOTH_F × (pid_roll  - smooth_roll_p)
  smooth_pitch_p += SMOOTH_F × (pid_pitch - smooth_pitch_p)

  smooth_roll  = clamp(smooth_roll_p,  -3.0, +3.0)
  smooth_pitch = clamp(smooth_pitch_p, -3.0, +3.0)
```

### Comportement du terme intégral en pratique

```
Scénario : piéton court à 0.875 m/s,
           drone bloqué à dist=0.5m (P seul insuffisant)

  t=0s  : P=0.375, I=0.000  → cmd=0.375 → drone à peine plus rapide que piéton
  t=1s  : P=0.375, I=0.175  → cmd=0.550 → drone accélère
  t=2s  : P=0.375, I=0.350  → cmd=0.725 → drone encore plus rapide
  t=3s  : P=0.375, I=0.525  → cmd=0.900 → drone dépasse légèrement le piéton
  t=3.5s: dist commence à diminuer → I commence à diminuer
  t=4s  : dist < 0.35m → [STABLE], I décroît × 0.90/frame
```

### Anti-windup lors de STABLE

Quand le drone est stable (`dist < 0.35m`), l'intégrale doit être vidée progressivement pour ne pas "sauter" quand le piéton repart :

```python
if dist_m < STABLE_DIST:
    integral_x *= 0.90     # décroît de 10% par frame YOLO
    integral_y *= 0.90     # → vidé à ~35% après 10 frames stables
    smooth_roll_p  *= 0.80
    smooth_pitch_p *= 0.80
    smooth_roll    *= 0.75
    smooth_pitch   *= 0.75
```

### Conventions de signe (vérifiées expérimentalement)

| Condition caméra | off_y | dy_m | pid_pitch | Effet moteurs | Mouvement drone |
|-----------------|-------|------|-----------|---------------|-----------------|
| Piéton en haut du frame | < 0 | < 0 | < 0 | FL/FR plus lents | Avance ✓ |
| Piéton en bas du frame | > 0 | > 0 | > 0 | RL/RR plus lents | Recule ✓ |
| Piéton à droite | off_x > 0 | dx_m > 0 | pid_roll < 0 | FR/RR plus lents | Strafe droite ✓ |
| Piéton à gauche | off_x < 0 | dx_m < 0 | pid_roll > 0 | FL/RL plus lents | Strafe gauche ✓ |

---

## 12. Machine d'états

```
                    ┌──────────┐
               ┌───►│ TAKEOFF  │ altitude < TARGET_ALT
               │    │  4.0m    │
               │    └────┬─────┘
               │         │ |alt - 4.0| < 0.15m
               │    ┌────▼─────┐
               │    │  SEARCH  │◄──────────────────────┐
               │    │  (hover) │                       │
               │    └────┬─────┘                       │
               │         │ YOLO détecte piéton         │
               │    ┌────▼──────────────────────┐      │
               │    │         FOLLOW            │      │
               │    │                           │      │
   alt < 1.5m  │    │  ┌────────────────────┐  │      │
               │    │  │ dist < 0.35m       │  │      │
               │    │  │ [STABLE]           │  │      │  lost_frames
               │    │  │ cmd → 0            │  │      │  ≥ 10
               │    │  ├────────────────────┤  │      │
               │    │  │ dist ≥ 0.35m       │  │      │
               │    │  │ [FOLLOW]           │  │      │
               │    │  │ PI control         │  │      │
               │    │  ├────────────────────┤  │      │
               │    │  │ YOLO perd cible    │  │──────┘
               │    │  │ [RECKONING] ×10    │  │
               │    │  └────────────────────┘  │
               └────┴───────────────────────────┘
```

### Transitions et conditions

| De | Vers | Condition |
|----|------|-----------|
| TAKEOFF | SEARCH | `|altitude - 4.0| < 0.15m` |
| SEARCH | FOLLOW | YOLO détecte au moins 1 boîte (conf > 0.40) |
| FOLLOW | SEARCH | `lost_frames ≥ 10` (piéton absent 10 frames consécutives) |
| FOLLOW | SEARCH | `altitude < 1.5m` (sécurité) |
| RECKONING | SEARCH | `lost_frames ≥ 10` |
| RECKONING | FOLLOW | YOLO re-détecte le piéton |

---

## 13. Dead reckoning

### Problème sans dead reckoning

Quand le piéton sort du champ de vision (bord du frame ou occlusion brève) :
- `YOLO` ne retourne aucune boîte
- Sans dead reckoning : smooth → 0 → drone **s'arrête immédiatement**
- Piéton continue à marcher → il s'éloigne encore plus → impossible de le retrouver

### Solution : maintenir la dernière commande connue

```
Piéton détecté frame N   : last_dx, last_dy sauvegardés
Piéton absent frame N+1  : lost_frames = 1
...
Piéton absent frame N+10 : lost_frames = 10 → retour SEARCH
```

Durant les 10 frames de reckoning, le drone continue avec la dernière erreur de position connue + l'intégrale encore chargée :

```python
dr_roll  = clamp(-last_dx * KP - integral_x * KI, -2.0, 2.0)
dr_pitch = clamp( last_dy * KP + integral_y * KI, -2.0, 2.0)
smooth_roll_p  += SMOOTH_F * (dr_roll  - smooth_roll_p)
smooth_pitch_p += SMOOTH_F * (dr_pitch - smooth_pitch_p)
```

**Durée du reckoning** : 10 frames × 8ms = **80ms** → pendant ce temps, un piéton à 0.875 m/s avance de 7cm supplémentaires, mais le drone continue dans la même direction et le reprend souvent.

---

## 14. Fenêtre de debug OpenCV

Une fenêtre temps réel (`cv2.imshow`) est affichée à chaque frame YOLO :

```python
def show_debug(raw_image, boxes, target, state_str, dist_m):
    img = np.frombuffer(raw_image, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ...
    cv2.imshow("Drone Camera", bgr)
    cv2.waitKey(1)
```

### Éléments visuels

| Élément | Couleur | Signification |
|---------|---------|---------------|
| Rectangle | Vert | Bounding box YOLO (tête détectée) |
| Croix | Jaune | Centre du frame = point cible |
| Point | Bleu | Centre tête quand `dist < 0.35m` (STABLE) |
| Point | Rouge | Centre tête quand `dist ≥ 0.35m` (FOLLOW) |
| Ligne | Même que point | Vecteur erreur : croix → tête |
| Texte | Cyan | État + distance en mètres |

La **ligne vecteur erreur** permet de visualiser instantanément dans quelle direction le drone doit se déplacer. Quand le point se superpose à la croix, le drone est parfaitement centré.

---

## 15. Comparaison des approches de contrôle testées

L'approche finale PI n'a pas été la première testée. Voici l'évolution chronologique :

### Approche 1 : Contrôleur P pur (pixels normalisés)

```python
target_yaw   = off_x * 2.0
target_pitch = off_y * 2.0 + abs(off_y) * off_y * 1.0  ← terme non-linéaire
target_roll  = -off_x * 1.5
```

**Résultat** : oscillation permanente (overshoot), le drone dépassait toujours le piéton.
**Problème** : gains trop élevés + terme non-linéaire amplifiant le dépassement.

### Approche 2 : PD avec feedforward vitesse

```python
t_roll  = -dx_m * KP - vx * KFF   # vx = d(off_x)/dt
t_pitch =  dy_m * KP + vy * KFF
```

**Résultat** : amélioration à faible vitesse, mais perte du piéton à 0.875 m/s.
**Problème** : `vx` était du bruit YOLO pur (signal/bruit < 1 avec téléportation piéton).

### Approche 3 : Feedforward split (immédiat + position lissée)

```python
ff_roll  = -vx * KFF                    # immédiat, sans filtre
smooth_roll_p += SMOOTH_F * (-dx_m * KP - smooth_roll_p)
smooth_roll  = ff_roll + smooth_roll_p  # total
```

**Résultat** : meilleur, mais instable à haute vitesse (bruit amplifié par KFF).
**Problème** : même nature — la vitesse estimée était du bruit.

### Approche 4 : PI pur avec intégrale à chaque frame ✅ (finale)

```python
integral += dx_m * dt_s       # accumulé à 8ms (chaque step)
cmd = -dx_m * KP - integral * KI
```

**Résultat** : suivi fiable à toutes les vitesses testées. La dist passe de 1.85m → 0.04m en ~70 frames (~1.1s).
**Avantage** : pas de dépendance à l'estimation de vitesse bruyante.

### Tableau comparatif

| Approche | Vitesse lente | Vitesse rapide | Oscillation | Complexité |
|----------|---------------|----------------|-------------|------------|
| P pur pixels | Médiocre | Non | Forte | Faible |
| PD + feedforward | Bon | Mauvais | Modérée | Moyenne |
| Feedforward split | Bon | Moyen | Faible | Haute |
| **PI frame (final)** | **Excellent** | **Bon** | **Nulle** | **Moyenne** |

---

## 16. Analyse de performance

### Trace de suivi typique (logs Webots)

```
[READY] Takeoff → 4.0m  dt=8ms
[STATE] 3.85m atteint → SEARCH
[STATE] Piéton détecté → FOLLOW

Phase 1 — Acquisition (piéton à 1.85m) :
[FOLLOW] dist=1.85m I=(+0.00,+0.00) | cmd=(-0.12,+0.68) | alt=4.64m
[FOLLOW] dist=1.30m I=(+0.16,+1.08) | cmd=(-0.19,+1.36) | alt=4.57m
[FOLLOW] dist=0.66m I=(+0.21,+1.41) | cmd=(-0.13,+1.01) | alt=4.51m

Phase 2 — Stabilisation :
[STABLE] dist=0.32m I=(+0.19,+1.35) | alt=4.48m
[STABLE] dist=0.07m I=(+0.08,+0.60) | alt=4.46m   ← 7cm d'erreur
[STABLE] dist=0.04m I=(+0.07,+0.54) | alt=4.45m   ← 4cm = quasi parfait

Phase 3 — Suivi (piéton repart) :
[FOLLOW] dist=0.36m I=(+0.03,+0.21) | cmd=(+0.03,-0.09) | alt=4.42m
[FOLLOW] dist=0.98m I=(-0.02,+0.02) | cmd=(+0.14,-0.69) | alt=4.36m
[FOLLOW] dist=1.70m I=(-0.12,-0.58) | cmd=(+0.26,-1.44) | alt=4.27m
```

### Métriques de performance extraites des logs

| Métrique | Valeur mesurée |
|----------|----------------|
| Temps acquisition (1.85m → STABLE) | ~1.1s (70 frames YOLO) |
| Erreur résiduelle en STABLE | **0.04m** (4 cm) |
| Commande max observée | ±2.50 |
| Intégrale max observée | ±1.49 |
| Fréquence détection YOLO | toutes les 16ms |
| Fréquence accumulation intégrale | toutes les 8ms |

---

## 17. Problèmes rencontrés et solutions

### P1 — YOLO centrait sur les épaules

| | Détail |
|-|--------|
| **Symptôme** | Drone se positionnait 20cm trop bas (sur les épaules) |
| **Cause racine** | Boîtes dataset couvrant tête+épaules → centre boîte = épaules |
| **Détection** | Fenêtre debug OpenCV : point rouge sur les épaules, pas la tête |
| **Solution** | Supprimer tous les `.txt`, re-labelliser uniquement la tête, re-entraîner |
| **Prévention** | Vérifier que `w` et `h` dans les `.txt` sont < 0.25 |

### P2 — Oscillation avant/arrière

| | Détail |
|-|--------|
| **Symptôme** | Drone oscillait en boucle : dépasse piéton → repart → dépasse → ... |
| **Cause racine** | `target_pitch = off_y * 2.0 + nonlinear` trop agressif |
| **Solution** | KP: 2.0 → 0.75, supprimer terme non-linéaire, SMOOTH_F = 0.50 |
| **Théorie** | Sans filtre, le PID réagit trop vite → overshoot → oscillation |

### P3 — Perte du piéton à vitesse élevée

| | Détail |
|-|--------|
| **Symptôme** | Drone perdait le piéton en ~1s dès 0.875 m/s |
| **Cause 1** | Feedforward vitesse = bruit (SNR < 1 avec piéton téléporté) |
| **Cause 2** | Intégrale accumulée trop lentement (aux frames YOLO seulement) |
| **Solution** | Supprimer KFF, accumuler intégrale à chaque frame (8ms), KI=0.35 |

### P4 — Labels JSON, pas YOLO

| | Détail |
|-|--------|
| **Symptôme** | `labels/train/*.txt` vide après labelling LabelMe |
| **Cause** | LabelMe utilise JSON, pas le format YOLO natif |
| **Solution** | Script Python de conversion + re-utiliser LabelImg à la place |

### P5 — Crash entraînement epoch 24

| | Détail |
|-|--------|
| **Symptôme** | Exit code 5 (Windows STATUS_ACCESS_VIOLATION) |
| **Cause** | Dépassement mémoire RAM avec batch=16 sur CPU |
| **Solution** | batch=8 + `model.train(resume=True)` depuis `last.pt` |

### P6 — `NameError: KFF` au runtime

| | Détail |
|-|--------|
| **Symptôme** | Crash ligne 322 au premier contact avec le dead reckoning |
| **Cause** | `KFF` supprimé des constantes mais encore référencé dans dead reckoning |
| **Solution** | Remplacer `vx_last * KFF` par `last_dx * KP + integral * KI` |

### P7 — Intégrale dans le mauvais sens après changement direction

| | Détail |
|-|--------|
| **Symptôme** | Après STABLE, le drone repartait trop vite dans le sens opposé |
| **Cause** | Intégrale chargée positivement → saut de commande lors du retour FOLLOW |
| **Solution** | `integral *= 0.90` à chaque frame STABLE → intégrale vidée proprement |

---

## 18. Paramètres finaux

### Contrôleur de suivi

```python
TARGET_ALT    = 4.0     # altitude de croisière (m)
STABLE_DIST   = 0.35    # seuil de stabilité (m)

KP            = 0.75    # gain proportionnel (m → disturbance)
KI            = 0.35    # gain intégral (accumulé à 8ms/step)
SMOOTH_F      = 0.50    # filtre passe-bas commande

YOLO_EVERY    = 2       # inférence YOLO toutes les 2 frames (16ms)
LOST_TOLERANE = 10      # frames dead reckoning avant SEARCH
```

### Stabilisation vol (PID original Webots C)

```python
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 3.0
K_ROLL_P          = 50.0
K_PITCH_P         = 30.0
```

### Caméra

```python
CAM_W    = 640          # pixels
CAM_H    = 480          # pixels
CAM_HFOV = 1.0          # radian (depuis .wbt)
CAM_VFOV = 0.775        # radian (calculé)
```

### Modèle YOLO

```
Fichier         : best.pt  (6.2 MB)
Architecture    : YOLOv8n (3M paramètres)
Classe          : 0 = person (tête vue de dessus)
Seuil confiance : 0.40
Inférence       : ~40ms / frame (CPU)
```

---

## 19. Limitations et travaux futurs

### Limitations actuelles

| Limitation | Description |
|------------|-------------|
| **Vitesse max** | Perte fréquente au-delà de 1.0 m/s (limite physique PID drone) |
| **Direction unique** | Piéton se déplace sur l'axe X uniquement (pas de virage) |
| **Pas de GPU** | Inférence YOLO à 40ms/frame sur CPU uniquement |
| **Dataset limité** | 353 images labellisées, environnement unique (sol Webots) |
| **Téléportation piéton** | Pas de physique réelle → pas d'estimation de vitesse fiable |
| **Yaw fixe** | Le drone ne tourne pas pour faire face au piéton |
| **Altitude fixe** | Pas d'adaptation dynamique de l'altitude selon la vitesse |

### Améliorations possibles

```
Court terme :
  ├─ Ajouter des virages au piéton (trajectoire en L, cercle, aléatoire)
  ├─ Augmenter le dataset (500+ images, variété de fonds)
  ├─ Implémenter le contrôle du yaw pour orienter le drone vers le piéton
  └─ Tester sur GPU pour réduire la latence YOLO à <10ms

Moyen terme :
  ├─ Altitude adaptive : monter à 5m quand vitesse détectée élevée
  ├─ Kalman filter pour estimer la position/vitesse piéton malgré le bruit
  ├─ Piéton physique (utiliser un contrôleur moteur Webots réaliste)
  └─ Multi-piétons : suivre la personne la plus proche

Long terme :
  ├─ Déploiement sur drone réel (ROS2 + caméra embarquée)
  ├─ Modèle YOLO amélioré : segmentation pour contour précis de la tête
  └─ Apprentissage par renforcement pour optimiser automatiquement KP, KI
```

---

## 20. Glossaire

| Terme | Définition |
|-------|-----------|
| **Webots** | Simulateur robotique open-source 3D développé par Cyberbotics, utilisé pour tester des algorithmes de contrôle sans risque physique |
| **YOLOv8** | *You Only Look Once v8* — réseau de neurones convolutif pour la détection d'objets en temps réel |
| **mAP50** | *Mean Average Precision at IoU=0.5* — métrique de qualité de détection YOLO |
| **IoU** | *Intersection over Union* — mesure du chevauchement entre boîte prédite et boîte réelle |
| **Recall** | Taux de détection correcte : TP / (TP + FN) |
| **Precision** | Taux de précision : TP / (TP + FP) |
| **PID** | *Proportional Integral Derivative* — contrôleur classique à trois termes |
| **PI** | Contrôleur PID sans terme dérivé (D = 0), utilisé ici |
| **Anti-windup** | Mécanisme limitant la croissance de l'intégrale pour éviter la saturation |
| **Dead reckoning** | Navigation par estimation de position sans capteur externe, basée sur la dernière mesure connue |
| **FOV** | *Field Of View* — champ de vue de la caméra |
| **BGRA** | Format d'image Webots : Blue, Green, Red, Alpha |
| **Supervisor** | API Webots permettant de contrôler la simulation directement (téléportation, modification de champs) |
| **off_x / off_y** | Offset normalisé [-1, 1] du centre de la tête par rapport au centre du frame caméra |
| **dx_m / dy_m** | Conversion de off_x/off_y en mètres réels via la géométrie caméra |
| **dist_m** | Distance euclidienne horizontale entre le drone et la tête du piéton |
| **SMOOTH_F** | Coefficient du filtre exponentiel sur les commandes (0 = aucun filtre, 1 = instantané) |
| **Jitter YOLO** | Variation pixel-à-pixel des bounding boxes entre deux frames consécutives, source de bruit |
| **Téléportation** | Déplacement du piéton par modification directe de sa position 3D (sans physique) |

---

## 21. Installation et lancement

### Prérequis

```bash
pip install ultralytics opencv-python numpy
```

Webots R2023b : téléchargeable sur [cyberbotics.com](https://cyberbotics.com)

### Lancer le projet

```
1. Ouvrir Webots
2. File → Open World → ps-drone/worlds/mavic_2_pro.wbt
3. Cliquer Play ▶
```

Le drone démarre automatiquement. Aucune intervention nécessaire.

### Modifier la vitesse du piéton

Dans `controllers/pedestrian_controller/pedestrian_controller.py` :

```python
new_pos = [pos[0] + 0.007, pos[1], pos[2]]
#                    ↑
#          0.002 → 0.25 m/s  (marche lente, suivi parfait)
#          0.005 → 0.625 m/s (marche normale)
#          0.007 → 0.875 m/s (marche rapide, suivi fiable)
#          0.010 → 1.25 m/s  (limite physique du drone)
```

### Contrôle manuel clavier

| Touche | Action |
|--------|--------|
| `↑` | Avancer |
| `↓` | Reculer |
| `←` | Rotation gauche (yaw) |
| `→` | Rotation droite (yaw) |
| `Shift + ←` | Strafe gauche |
| `Shift + →` | Strafe droite |
| `Shift + ↑` | Augmenter altitude cible (+5cm) |
| `Shift + ↓` | Diminuer altitude cible (-5cm) |

> Le contrôle manuel et le suivi autonome coexistent : les perturbations clavier s'ajoutent aux commandes PI. Il est possible de corriger manuellement le drone pendant le suivi.

---

*Projet réalisé dans le cadre d'une simulation de drone autonome — Webots + YOLOv8 + Contrôle PI*
