"""
capture_controller.py
Flies to a series of waypoints at multiple altitudes and saves frames.

HOW TO USE:
  1. In mavic_2_pro.wbt change Mavic2Pro's controller to "capture_controller"
  2. Run the simulation — it flies, stabilises, and saves automatically
  3. When the console prints "CAPTURE COMPLETE", stop and switch back
  4. Label the images in dataset/images/altitude_captures/ and retrain
"""

from controller import Robot, GPS, InertialUnit, Gyro, Camera
import math, os
import numpy as np
import cv2

# ── SAVE DIRECTORY ─────────────────────────────────────────────────────────────
HERE     = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.normpath(os.path.join(HERE, '..', '..', 'dataset', 'images', 'altitude_captures'))
os.makedirs(SAVE_DIR, exist_ok=True)

# ── MISSION PARAMETERS ────────────────────────────────────────────────────────
# Pedestrian starts near (3, 0).
# At low altitude the camera FOV is small (~2m wide at 2m height) so
# waypoints stay within ~1.5m of the pedestrian to keep them in frame.
WAYPOINTS = [
    ( 3.0,  0.0),   # directly above pedestrian
    ( 4.0,  0.0),   # 1 m ahead
    ( 2.0,  0.0),   # 1 m behind
    ( 3.0,  1.2),   # 1.2 m to the right
    ( 3.0, -1.2),   # 1.2 m to the left
    ( 4.2,  1.0),   # diagonal forward-right
    ( 1.8, -1.0),   # diagonal back-left
    ( 4.0, -1.2),   # diagonal forward-left
]

# Low altitudes — same range the drone uses in follow/stabilise mode
ALTITUDES = [1.5, 2.0, 2.5, 3.0]
FRAMES_PER_STOP  = 6     # frames per stop (more frames, more variety)
STABILISE_SECS   = 2.0   # a bit longer — more important to be still at low alt
WAYPOINT_THRESH  = 0.5   # tighter threshold at low altitude
ALT_THRESH       = 0.2   # tighter altitude tolerance

# ── POSITION CONTROL GAINS ────────────────────────────────────────────────────
# Smaller MAX_POS at low altitude — safer, no risk of hitting the ground
KP_POS   = 0.40
MAX_POS  = 1.0    # conservative — drone moves slowly

# ── FLIGHT CONSTANTS (same as main controller) ────────────────────────────────
K_VERTICAL_THRUST = 68.5
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P      = 4.0
K_ROLL_P          = 50.0
K_PITCH_P         = 30.0

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# ── ROBOT SETUP ───────────────────────────────────────────────────────────────
robot    = Robot()
timestep = int(robot.getBasicTimeStep())
dt_s     = timestep / 1000.0

camera = robot.getDevice("camera"); camera.enable(timestep)
CAM_W  = camera.getWidth()
CAM_H  = camera.getHeight()

imu  = robot.getDevice("inertial unit"); imu.enable(timestep)
gps  = robot.getDevice("gps");          gps.enable(timestep)
gyro = robot.getDevice("gyro");         gyro.enable(timestep)

cam_roll_motor  = robot.getDevice("camera roll")
cam_pitch_motor = robot.getDevice("camera pitch")
front_left_led  = robot.getDevice("front left led")
front_right_led = robot.getDevice("front right led")

motors = [
    robot.getDevice("front left propeller"),
    robot.getDevice("front right propeller"),
    robot.getDevice("rear left propeller"),
    robot.getDevice("rear right propeller"),
]
for m in motors:
    m.setPosition(float('inf'))
    m.setVelocity(1.0)

# ── MISSION STATE ─────────────────────────────────────────────────────────────
alt_idx     = 0
wp_idx      = 0
phase       = "fly_to"   # "fly_to" → "stabilise" → "capture" → next wp
stab_start  = 0.0
cap_count   = 0
total_saved = 0

TARGET_ALT = ALTITUDES[alt_idx]

total_stops = len(ALTITUDES) * len(WAYPOINTS)
print(f"[CAPTURE] Saving to: {SAVE_DIR}")
print(f"[CAPTURE] {len(ALTITUDES)} altitudes × {len(WAYPOINTS)} waypoints × "
      f"{FRAMES_PER_STOP} frames = {total_stops * FRAMES_PER_STOP} total frames")
print(f"[CAPTURE] Starting → alt={TARGET_ALT}m  wp={WAYPOINTS[0]}")

# ── PREVIEW ────────────────────────────────────────────────────────────────────
def show_preview(raw, alt, wp, phase_str, saved, total):
    img = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    cv2.putText(bgr, f"alt={alt:.1f}m  wp={wp}  [{phase_str}]  {saved}/{FRAMES_PER_STOP}  total={total}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 180), 1)
    cv2.imshow("Dataset Capture", bgr)
    cv2.waitKey(1)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
while robot.step(timestep) != -1:
    t        = robot.getTime()
    roll     = imu.getRollPitchYaw()[0]
    pitch    = imu.getRollPitchYaw()[1]
    yaw      = imu.getRollPitchYaw()[2]
    altitude = gps.getValues()[2]
    cx, cy   = gps.getValues()[0], gps.getValues()[1]
    roll_v   = gyro.getValues()[0]
    pitch_v  = gyro.getValues()[1]

    # Camera stabilisation + LEDs
    cam_roll_motor.setPosition(clamp(-0.115 * roll_v,  -0.5, 0.5))
    cam_pitch_motor.setPosition(clamp(-0.1   * pitch_v, -0.5, 0.5))
    led = int(t) % 2
    front_left_led.set(led); front_right_led.set(1 - led)

    # Current waypoint target
    tx, ty = WAYPOINTS[wp_idx]

    # ── HORIZONTAL POSITION CONTROL ──────────────────────────────────────────
    # World-frame error
    ex = tx - cx
    ey = ty - cy
    dist_xy = math.sqrt(ex**2 + ey**2)

    # Rotate error into drone body frame using IMU yaw.
    # At yaw=0 (drone faces +X):  body_fwd = ex, body_left = ey
    body_fwd  = math.cos(yaw) * ex + math.sin(yaw) * ey
    body_left = -math.sin(yaw) * ex + math.cos(yaw) * ey

    # pitch_d < 0 → forward (+X); roll_d > 0 → strafe left (+Y)
    roll_d  = clamp( KP_POS * body_left, -MAX_POS, MAX_POS)
    pitch_d = clamp(-KP_POS * body_fwd,  -MAX_POS, MAX_POS)

    # ── PHASE MACHINE ─────────────────────────────────────────────────────────
    at_pos = dist_xy < WAYPOINT_THRESH and abs(altitude - TARGET_ALT) < ALT_THRESH

    if phase == "fly_to":
        if at_pos:
            phase      = "stabilise"
            stab_start = t
            print(f"[CAPTURE] Reached wp={WAYPOINTS[wp_idx]} alt={TARGET_ALT:.1f}m — stabilising...")

    elif phase == "stabilise":
        roll_d  *= 0.3   # damp lateral movement during stabilisation
        pitch_d *= 0.3
        if t - stab_start >= STABILISE_SECS:
            phase     = "capture"
            cap_count = 0

    elif phase == "capture":
        roll_d  *= 0.2   # nearly hover while snapping frames
        pitch_d *= 0.2

        raw = camera.getImage()
        if raw:
            show_preview(raw, TARGET_ALT, WAYPOINTS[wp_idx], "CAPTURE", cap_count, total_saved)

            if cap_count < FRAMES_PER_STOP:
                img = np.frombuffer(raw, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
                bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                fname = f"alt{TARGET_ALT:.1f}m_wp{wp_idx:02d}_{cap_count:03d}.jpg"
                cv2.imwrite(os.path.join(SAVE_DIR, fname), bgr)
                cap_count   += 1
                total_saved += 1
                print(f"[CAPTURE] {fname}  ({cap_count}/{FRAMES_PER_STOP})")
            else:
                # Advance to next waypoint / altitude
                wp_idx += 1
                if wp_idx >= len(WAYPOINTS):
                    wp_idx  = 0
                    alt_idx += 1
                    if alt_idx >= len(ALTITUDES):
                        print(f"\n[CAPTURE] COMPLETE — {total_saved} frames saved to:")
                        print(f"  {SAVE_DIR}")
                        print("Stop the simulation, label the images, then retrain.")
                        break
                    TARGET_ALT = ALTITUDES[alt_idx]
                    print(f"[CAPTURE] ── new altitude {TARGET_ALT}m ──")

                print(f"[CAPTURE] → wp={WAYPOINTS[wp_idx]}  alt={TARGET_ALT:.1f}m")
                phase = "fly_to"

    # ── MOTOR COMMANDS ────────────────────────────────────────────────────────
    roll_input  = K_ROLL_P  * clamp(roll,  -1.0, 1.0) + roll_v  + roll_d
    pitch_input = K_PITCH_P * clamp(pitch, -1.0, 1.0) + pitch_v + pitch_d

    clamped_diff_alt = clamp(TARGET_ALT - altitude + K_VERTICAL_OFFSET, -1.0, 1.0)
    vertical_input   = K_VERTICAL_P * (clamped_diff_alt ** 3)

    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input

    motors[0].setVelocity( fl)
    motors[1].setVelocity(-fr)
    motors[2].setVelocity(-rl)
    motors[3].setVelocity( rr)
