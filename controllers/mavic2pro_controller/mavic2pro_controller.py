from controller import Robot, Compass, GPS, Gyro, InertialUnit, Keyboard, LED, Motor
import math
import numpy as np
import cv2
from ultralytics import YOLO

# ── MODEL ─────────────────────────────────────────────────────────────────────
print("[INIT] Loading YOLO model...")
model = YOLO(r"C:\Users\Maryem\Desktop\ps-drone\controllers\mavic2pro_controller\best.pt")
print("[INIT] YOLO ready.")

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
TARGET_ALT  = 4.0    # altitude cible — 4m donne un demi-champ de 2.18m
                     # à 2 m/s le piéton prend 1.09s à sortir du frame (vs 0.82s à 3m)
STABLE_DIST = 0.35   # seuil stabilité (mètres)

# FOV caméra (depuis mavic_2_pro.wbt : fieldOfView 1)
CAM_HFOV = 1.0                                       # radians (≈ 57°)
CAM_VFOV = 2 * math.atan(math.tan(CAM_HFOV / 2)     # dérivé depuis le ratio
                         * CAM_H / CAM_W)            # ≈ 0.775 rad (≈ 44°)

# ── TRACKING PARAMETERS ───────────────────────────────────────────────────────
YOLO_EVERY    = 2
LOST_TOLERANE = 20  # was 10 — gives 320ms recovery window at dt=8ms

KP          = 1.20   # reduced: less oscillation around target
KI          = 0.50
KD          = 0.10
VEL_ALPHA   = 0.20   # more filtering: YOLO box noise is smoothed over more frames
VEL_RAW_MAX = 4.0
SMOOTH_F    = 0.55   # slower command transitions → less visible jitter

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
    results = model(rgb, verbose=False, classes=[0])
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        if conf > 0.40:
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
        key = kb.getKey()

    frame_ctr += 1

    # ── MACHINE D'ÉTATS ───────────────────────────────────────────────────────

    if state == "takeoff":
        smooth_roll = smooth_pitch = smooth_yaw = 0.0
        smooth_roll_p = smooth_pitch_p = 0.0
        if abs(altitude - TARGET_ALT) < 0.15:
            state = "search"
            print(f"[STATE] {altitude:.2f}m atteint → SEARCH")

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
                print("[STATE] Piéton détecté → FOLLOW")

    elif state == "follow":
        # Sécurité altitude
        if altitude < 1.5:
            smooth_roll = smooth_pitch = smooth_yaw = 0.0
            state = "search"
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
                    # EMA update only here (far from target = reliable signal).
                    # prev_dx = None after any detection gap → first frame after
                    # reckoning has no vel update, preventing velocity explosion.
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

                    pid_roll  = clamp(-dx_m * KP - integral_x * KI - vel_x * KD, -2.5, 2.5)
                    pid_pitch = clamp( dy_m * KP + integral_y * KI + vel_y * KD, -2.5, 2.5)

                    smooth_roll_p  += SMOOTH_F * (pid_roll  - smooth_roll_p)
                    smooth_pitch_p += SMOOTH_F * (pid_pitch - smooth_pitch_p)

                    smooth_roll  = clamp(smooth_roll_p,  -2.5, 2.5)
                    smooth_pitch = clamp(smooth_pitch_p, -2.5, 2.5)
                    smooth_yaw  *= 0.9

                    print(f"[FOLLOW] dist={dist_m:.2f}m "
                          f"vel=({vel_x:+.2f},{vel_y:+.2f}) | "
                          f"cmd=({smooth_roll:+.2f},{smooth_pitch:+.2f}) | "
                          f"alt={altitude:.2f}m")

                show_debug(raw, boxes, target, "FOLLOW", dist_m)

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
                    # Commands decay toward zero as reckoning lengthens — prevents
                    # the drone from flying at max tilt for 20 frames and losing altitude.
                    fade = 0.92 ** (lost_frames - 1)
                    dr_roll  = clamp((-last_dx * KP - integral_x * KI - vel_x * KD) * fade, -1.5, 1.5)
                    dr_pitch = clamp(( last_dy * KP + integral_y * KI + vel_y * KD) * fade, -1.5, 1.5)
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
                    print("[STATE] Piéton perdu → SEARCH")

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
