import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'

from controller import Robot, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import math
import numpy as np
import cv2
from ultralytics import YOLO

# ── MODEL ─────────────────────────────────────────────────────────────────────
print("[INIT] Loading YOLO model...")
model = YOLO(r"C:\Users\Maryem\Desktop\ps-drone\controllers\mavic2pro_controller\best.pt")
print("[INIT] YOLO ready.")

# ── LIVE CAPTURE ──────────────────────────────────────────────────────────────
# Appuie sur 'C' pendant la simulation pour activer/désactiver la sauvegarde.
# Les images sont sauvegardées avec leur annotation YOLO (.txt) — prêtes à
# l'emploi pour le ré-entraînement sans labellisation manuelle.
_HERE      = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR   = os.path.normpath(os.path.join(_HERE, '..', '..', 'dataset', 'images', 'live_captures'))
os.makedirs(SAVE_DIR, exist_ok=True)

AUTO_SAVE      = False   # toggle avec la touche 'C'
SAVE_EVERY     = 15      # sauvegarde 1 frame toutes les N frames YOLO détectées
_save_ctr      = 0
_saved_total   = 0
_c_was_pressed = False   # edge-detector : évite le double-toggle sur key-repeat
print(f"[CAPTURE] Appuie sur 'C' pour activer la capture live → {SAVE_DIR}")

def clamp(val, lo, hi):
    return max(lo, min(val, hi))

# ── PID CONSTANTS (original C code values) ────────────────────────────────────
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 4.0   # 4.0: strong hold without takeoff overshoot
K_ROLL_P          = 50.0
K_PITCH_P         = 30.0

# ── ROBOT & DEVICES ───────────────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice("camera")
camera.enable(timestep)
CAM_W = camera.getWidth()    # 640
CAM_H = camera.getHeight()   # 480
print(f"[INIT] Camera {CAM_W}x{CAM_H}")

imu     = robot.getDevice("inertial unit"); imu.enable(timestep)
gps     = robot.getDevice("gps");          gps.enable(timestep)
compass = robot.getDevice("compass");      compass.enable(timestep)
gyro    = robot.getDevice("gyro");         gyro.enable(timestep)
kb      = robot.getKeyboard();             kb.enable(timestep)

front_left_led  = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")
cam_roll_motor  = robot.getDevice("camera roll")
cam_pitch_motor = robot.getDevice("camera pitch")

motors = [
    robot.getDevice("front left propeller"),
    robot.getDevice("front right propeller"),
    robot.getDevice("rear left propeller"),
    robot.getDevice("rear right propeller"),
]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(1.0)

# ── FLIGHT PARAMETERS ─────────────────────────────────────────────────────────
SEARCH_ALT    = 4.0    # altitude en mode recherche — FOV 1.8 rad couvre ~10m à 4m
FOLLOW_ALT    = 2.3    # altitude parapluie — ~0.5m au-dessus de la tête du piéton
TARGET_ALT    = SEARCH_ALT
STABLE_DIST   = 0.35   # seuil stabilité (mètres)
DESCENT_RATE  = 0.5    # m/s — vitesse de descente progressive vers le piéton

# FOV caméra (depuis mavic_2_pro.wbt : fieldOfView 1)
CAM_HFOV = 1.8                                       # radians (≈ 103°) — grand angle pour suivi proche
CAM_VFOV = 2 * math.atan(math.tan(CAM_HFOV / 2)     # dérivé depuis le ratio
                         * CAM_H / CAM_W)            # ≈ 0.775 rad (≈ 44°)

# ── TRACKING PARAMETERS ───────────────────────────────────────────────────────
YOLO_EVERY    = 2
LOST_TOLERANE = 20  # was 10 — gives 320ms recovery window at dt=8ms

KP          = 1.60   # plus fort : réduit l'erreur de position résiduelle
KI          = 0.06   # très bas : évite le windup intégral
KD          = 0.45   # plus élevé : anticipe mieux le mouvement du piéton
VEL_ALPHA   = 0.35   # mis à jour plus vite : vitesse estimée réagit aux changements
VEL_RAW_MAX = 4.0
SMOOTH_F    = 0.70   # plus réactif : divise par 2 le lag sur les changements de direction

# ── STATE ─────────────────────────────────────────────────────────────────────
state      = "takeoff"
frame_ctr  = 0
dt_s       = 0.0   # sera mis à jour depuis timestep

smooth_roll_p  = 0.0
smooth_pitch_p = 0.0
smooth_roll    = 0.0
smooth_pitch   = 0.0
smooth_yaw     = 0.0

last_dx    = 0.0   # dernière erreur connue (mise à jour chaque YOLO)
last_dy    = 0.0
integral_x = 0.0   # intégrale accumulée à CHAQUE frame
integral_y = 0.0

lost_frames  = 0
prev_dx = prev_dy = None   # last YOLO dx/dy — used for velocity estimation
vel_x   = 0.0              # EMA-filtered velocity of the error (m/s)
vel_y   = 0.0


# ── FONCTIONS YOLO ────────────────────────────────────────────────────────────
def detect(raw_image):
    """YOLO sur l'image brute. Retourne liste de (x1,y1,x2,y2,conf)."""
    img = np.frombuffer(raw_image, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
    rgb = img[:, :, :3][:, :, ::-1].copy()
    # imgsz=640 : on garde la résolution native — à 6m la personne fait ~60px,
    # si on laisse YOLO réduire à 320 elle ne fait plus que ~30px (limite de détection).
    results = model(rgb, verbose=False, classes=[0], imgsz=640)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        if conf > 0.25:   # seuil abaissé : 0.40 → 0.25 pour capturer les détections faibles
            boxes.append((float(x1), float(y1), float(x2), float(y2), conf))
    return boxes


def best_box(boxes):
    """Sélectionne la plus grande boîte (piéton le plus proche/confiant)."""
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


# ── FORMULES GÉOMÉTRIQUES ─────────────────────────────────────────────────────
def world_displacement(box, altitude):
    """
    Convertit le centre de la boîte en déplacement réel (mètres).

    Principe :
      Le drone est à `altitude` mètres au-dessus du sol, caméra pointant vers le bas.
      L'offset normalisé [-1, 1] × le demi-champ de vue donne l'angle,
      et altitude × tan(angle) donne la distance réelle au sol.

      dx_m = off_x × altitude × tan(HFOV/2)
      dy_m = off_y × altitude × tan(VFOV/2)

    Retourne (dx_m, dy_m) :
      dx_m > 0 → piéton à DROITE du drone
      dy_m > 0 → piéton en BAS du frame (drone doit reculer)
      dx_m = dy_m = 0 → drone est exactement au-dessus
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0

    off_x = (cx - CAM_W / 2.0) / (CAM_W / 2.0)  # [-1, 1]
    off_y = (cy - CAM_H / 2.0) / (CAM_H / 2.0)  # [-1, 1]

    dx_m = off_x * altitude * math.tan(CAM_HFOV / 2.0)
    dy_m = off_y * altitude * math.tan(CAM_VFOV / 2.0)
    return dx_m, dy_m


# ── DEBUG WINDOW ─────────────────────────────────────────────────────────────
def show_debug(raw_image, boxes, target, state_str, dist_m):
    img = np.frombuffer(raw_image, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    for b in boxes:
        cv2.rectangle(bgr,
                      (int(b[0]), int(b[1])), (int(b[2]), int(b[3])),
                      (0, 200, 0), 2)

    # Croix au centre du frame (où le drone doit maintenir la tête)
    cv2.drawMarker(bgr, (CAM_W // 2, CAM_H // 2),
                   (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    if target is not None:
        cx = int((target[0] + target[2]) / 2)
        cy = int((target[1] + target[3]) / 2)
        # Bleu si stable, rouge si en train de suivre
        color = (255, 100, 0) if dist_m < STABLE_DIST else (0, 80, 255)
        cv2.circle(bgr, (cx, cy), 7, color, -1)
        # Ligne entre centre du frame et centre de la tête
        cv2.line(bgr, (CAM_W // 2, CAM_H // 2), (cx, cy), color, 1)

    label = f"{state_str}  dist={dist_m:.2f}m" if dist_m >= 0 else state_str
    cv2.putText(bgr, label, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.imshow("Drone Camera", bgr)
    cv2.waitKey(1)


# ── LIVE CAPTURE HELPER ───────────────────────────────────────────────────────
def _save_frame(raw_image, altitude):
    """Sauvegarde l'image brute pour labellisation manuelle."""
    global _saved_total
    img  = np.frombuffer(raw_image, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr  = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    stem = f"live_alt{altitude:.1f}m_{_saved_total:05d}"
    cv2.imwrite(os.path.join(SAVE_DIR, f"{stem}.jpg"), bgr)
    _saved_total += 1
    print(f"[CAPTURE] {stem}.jpg  (total: {_saved_total})")


# ── STABILISATION INITIALE ────────────────────────────────────────────────────
dt_s = timestep / 1000.0   # pas de simulation en secondes (ex: 8ms → 0.008)

print("[INIT] Stabilising...")
while robot.step(timestep) != -1:
    if robot.getTime() > 1.0:
        break
print(f"[READY] Takeoff → {TARGET_ALT}m  dt={dt_s*1000:.0f}ms")


# ── BOUCLE PRINCIPALE ─────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    t        = robot.getTime()
    roll     = imu.getRollPitchYaw()[0]
    pitch    = imu.getRollPitchYaw()[1]
    altitude = gps.getValues()[2]
    roll_v   = gyro.getValues()[0]
    pitch_v  = gyro.getValues()[1]

    # LEDs clignotantes
    led = int(t) % 2
    front_left_led.set(led)
    front_right_led.set(1 - led)

    # Stabilisation mécanique de la caméra
    cam_roll_motor.setPosition(clamp(-0.115 * roll_v,  -0.5, 0.5))
    cam_pitch_motor.setPosition(clamp(-0.1   * pitch_v, -0.5, 0.5))

    # ── CLAVIER (contrôle manuel) ─────────────────────────────────────────────
    roll_d = pitch_d = yaw_d = 0.0
    c_pressed_now = False
    key = kb.getKey()
    while key > 0:
        if   key == kb.UP:             pitch_d = -2.0
        elif key == kb.DOWN:           pitch_d =  2.0
        elif key == kb.RIGHT:          yaw_d   = -1.3
        elif key == kb.LEFT:           yaw_d   =  1.3
        elif key == kb.SHIFT+kb.RIGHT: roll_d  = -1.0
        elif key == kb.SHIFT+kb.LEFT:  roll_d  =  1.0
        elif key == kb.SHIFT+kb.UP:
            TARGET_ALT += 0.05
            print(f"[ALT] → {TARGET_ALT:.2f}m")
        elif key == kb.SHIFT+kb.DOWN:
            TARGET_ALT -= 0.05
            print(f"[ALT] → {TARGET_ALT:.2f}m")
        elif key == ord('C'):
            c_pressed_now = True
        key = kb.getKey()

    # Toggle uniquement sur le front montant (pressé → relâché)
    if c_pressed_now and not _c_was_pressed:
        AUTO_SAVE = not AUTO_SAVE
        print(f"[CAPTURE] {'ACTIVÉE' if AUTO_SAVE else 'DÉSACTIVÉE'}  "
              f"(total sauvegardé : {_saved_total})")
    _c_was_pressed = c_pressed_now

    frame_ctr += 1

    # ── MACHINE D'ÉTATS ───────────────────────────────────────────────────────

    if state == "takeoff":
        smooth_roll = smooth_pitch = smooth_yaw = 0.0
        smooth_roll_p = smooth_pitch_p = 0.0
        if abs(altitude - TARGET_ALT) < 0.15:
            state = "search"
            TARGET_ALT = SEARCH_ALT
            print(f"[STATE] {altitude:.2f}m atteint → SEARCH (montée à {SEARCH_ALT}m)")

    elif state == "search":
        smooth_roll = smooth_pitch = smooth_yaw = 0.0
        smooth_roll_p = smooth_pitch_p = 0.0
        integral_x = integral_y = 0.0
        last_dx = last_dy = 0.0
        raw = camera.getImage()
        if raw and frame_ctr % YOLO_EVERY == 0:
            boxes = detect(raw)
            show_debug(raw, boxes, None, "SEARCH", -1)
            if boxes:
                lost_frames = 0
                prev_dx = prev_dy = None
                vel_x = vel_y = 0.0
                state = "follow"
                # Ne pas changer TARGET_ALT ici : la descente se fait
                # progressivement dans follow, uniquement quand le piéton
                # est visible, pour garder le champ de vision sur lui.
                print(f"[STATE] Piéton détecté → FOLLOW (descente progressive {altitude:.1f}m → {FOLLOW_ALT}m)")

    elif state == "follow":
        # Sécurité altitude
        if altitude < 2.0:
            smooth_roll = smooth_pitch = smooth_yaw = 0.0
            state = "search"
            TARGET_ALT = SEARCH_ALT
            print(f"[SAFETY] Trop bas ({altitude:.2f}m) → SEARCH")

        raw = camera.getImage()
        if raw and frame_ctr % YOLO_EVERY == 0:
            boxes  = detect(raw)
            target = best_box(boxes)

            if target is not None:
                lost_frames = 0

                # ── CALCUL DU DÉPLACEMENT RÉEL ───────────────────────────
                dx_m, dy_m = world_displacement(target, altitude)
                dist_m     = math.sqrt(dx_m**2 + dy_m**2)
                last_dx, last_dy = dx_m, dy_m

                yolo_dt = YOLO_EVERY * dt_s

                # ── DESCENTE PROGRESSIVE ─────────────────────────────────
                # On descend uniquement quand le piéton est visible.
                # Si on le perd pendant la descente, on maintient l'altitude
                # et on reprend dès qu'il réapparaît.
                if TARGET_ALT > FOLLOW_ALT:
                    TARGET_ALT = max(FOLLOW_ALT,
                                     TARGET_ALT - DESCENT_RATE * YOLO_EVERY * dt_s)
                    if TARGET_ALT == FOLLOW_ALT:
                        print(f"[STATE] Altitude de suivi {FOLLOW_ALT}m atteinte")

                # ── DÉCISION : STABLE ou FOLLOW ──────────────────────────
                if dist_m < STABLE_DIST:
                    # ── STABLE : target close — decay vel (noisy), gentle PI ──
                    # EMA update is skipped here: near-target detections are
                    # noisy and the drone may overshoot; decaying vel prevents
                    # the KD term from growing and destabilising.
                    vel_x *= 0.55
                    vel_y *= 0.55
                    prev_dx, prev_dy = dx_m, dy_m

                    pid_roll  = clamp(-dx_m * KP * 0.35 - integral_x * KI * 0.35, -1.5, 1.5)
                    pid_pitch = clamp( dy_m * KP * 0.35 + integral_y * KI * 0.35, -1.5, 1.5)
                    smooth_roll_p  += SMOOTH_F * (pid_roll  - smooth_roll_p)
                    smooth_pitch_p += SMOOTH_F * (pid_pitch - smooth_pitch_p)
                    smooth_roll  = clamp(smooth_roll_p, -1.5, 1.5)
                    smooth_pitch = clamp(smooth_pitch_p, -1.5, 1.5)
                    smooth_yaw  *= 0.9
                    print(f"[STABLE] dist={dist_m:.2f}m "
                          f"vel=({vel_x:+.2f},{vel_y:+.2f}) | "
                          f"alt={altitude:.2f}m")

                else:
                    # ── FOLLOW : PID + velocity feedforward ───────────────
                    if prev_dx is not None:
                        raw_vx = (dx_m - prev_dx) / yolo_dt
                        raw_vy = (dy_m - prev_dy) / yolo_dt
                        raw_speed = math.sqrt(raw_vx**2 + raw_vy**2)
                        if raw_speed > VEL_RAW_MAX:
                            scale = VEL_RAW_MAX / raw_speed
                            raw_vx *= scale
                            raw_vy *= scale
                        vel_x = VEL_ALPHA * raw_vx + (1.0 - VEL_ALPHA) * vel_x
                        vel_y = VEL_ALPHA * raw_vy + (1.0 - VEL_ALPHA) * vel_y
                    prev_dx, prev_dy = dx_m, dy_m

                    # Frein de proximité : vitesse max proportionnelle à la distance.
                    # Quand le drone s'approche du piéton, il ralentit automatiquement
                    # et ne peut plus le dépasser même si l'intégrale a du windup.
                    max_cmd = clamp(dist_m * 2.5, 0.4, 2.5)

                    pid_roll  = clamp(-dx_m * KP - integral_x * KI - vel_x * KD, -max_cmd, max_cmd)
                    pid_pitch = clamp( dy_m * KP + integral_y * KI + vel_y * KD, -max_cmd, max_cmd)

                    smooth_roll_p  += SMOOTH_F * (pid_roll  - smooth_roll_p)
                    smooth_pitch_p += SMOOTH_F * (pid_pitch - smooth_pitch_p)

                    smooth_roll  = clamp(smooth_roll_p,  -max_cmd, max_cmd)
                    smooth_pitch = clamp(smooth_pitch_p, -max_cmd, max_cmd)
                    smooth_yaw  *= 0.9

                    print(f"[FOLLOW] dist={dist_m:.2f}m max_cmd={max_cmd:.2f} "
                          f"vel=({vel_x:+.2f},{vel_y:+.2f}) | "
                          f"cmd=({smooth_roll:+.2f},{smooth_pitch:+.2f}) | "
                          f"alt={altitude:.2f}m")

                show_debug(raw, boxes, target, "FOLLOW", dist_m)

                # ── LIVE CAPTURE ─────────────────────────────────────────
                if AUTO_SAVE:
                    _save_ctr += 1
                    if _save_ctr % SAVE_EVERY == 0:
                        _save_frame(raw, altitude)

            else:
                # ── DEAD RECKONING : piéton hors frame ───────────────────
                lost_frames += 1

                # Only reset prev_dx after 3+ missed frames.
                # For 1-2 frame gaps (YOLO glitch), keep prev_dx so velocity
                # estimation resumes cleanly on re-detection.
                if lost_frames >= 3:
                    prev_dx = prev_dy = None

                # Partial integral decay on first miss — prevents windup
                # from building up during a long reckoning stretch.
                if lost_frames == 1:
                    integral_x *= 0.70
                    integral_y *= 0.70

                show_debug(raw, boxes, None, f"RECKONING({lost_frames}/{LOST_TOLERANE})", -1)

                if lost_frames < LOST_TOLERANE:
                    # Prédit où est le piéton maintenant en prolongeant sa vitesse.
                    # Si le piéton marchait vers la droite quand la détection s'est
                    # perdue, il est probablement encore plus à droite maintenant.
                    fade   = 0.92 ** (lost_frames - 1)
                    lost_t = lost_frames * YOLO_EVERY * dt_s
                    pred_dx = clamp(last_dx + vel_x * lost_t, -3.0, 3.0)
                    pred_dy = clamp(last_dy + vel_y * lost_t, -3.0, 3.0)
                    dr_roll  = clamp((-pred_dx * KP - integral_x * KI - vel_x * KD) * fade, -1.5, 1.5)
                    dr_pitch = clamp(( pred_dy * KP + integral_y * KI + vel_y * KD) * fade, -1.5, 1.5)
                    smooth_roll_p  += SMOOTH_F * (dr_roll  - smooth_roll_p)
                    smooth_pitch_p += SMOOTH_F * (dr_pitch - smooth_pitch_p)
                    smooth_roll  = clamp(smooth_roll_p,  -1.5, 1.5)
                    smooth_pitch = clamp(smooth_pitch_p, -1.5, 1.5)
                    print(f"[RECKONING] cmd=({smooth_roll:+.2f},{smooth_pitch:+.2f})")
                else:
                    smooth_roll  *= 0.5
                    smooth_pitch *= 0.5
                    smooth_yaw   *= 0.5
                    lost_frames   = 0
                    state         = "search"
                    TARGET_ALT    = SEARCH_ALT
                    print(f"[STATE] Piéton perdu → SEARCH (remontée à {SEARCH_ALT}m)")

    # ── INTÉGRALE ACCUMULÉE À CHAQUE FRAME ────────────────────────────────────
    if state == "follow":
        integral_x = clamp(integral_x + last_dx * dt_s, -5.0, 5.0)
        integral_y = clamp(integral_y + last_dy * dt_s, -5.0, 5.0)

    # ── AJOUT DES PERTURBATIONS DE SUIVI AUX PERTURBATIONS CLAVIER ───────────
    roll_d  += smooth_roll
    pitch_d += smooth_pitch
    yaw_d   += smooth_yaw

    # ── PID STABILISATION (identique au code C original) ─────────────────────
    roll_input  = K_ROLL_P  * clamp(roll,  -1.0, 1.0) + roll_v  + roll_d
    pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_v + pitch_d
    yaw_input   = yaw_d

    clamped_diff_alt = clamp(TARGET_ALT - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
    vertical_input   = K_VERTICAL_P * (clamped_diff_alt ** 3)

    # Commandes moteurs (identique au code C)
    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input

    motors[0].setVelocity( fl)   # avant gauche
    motors[1].setVelocity(-fr)   # avant droit  (contre-rotation)
    motors[2].setVelocity(-rl)   # arrière gauche (contre-rotation)
    motors[3].setVelocity( rr)   # arrière droit
