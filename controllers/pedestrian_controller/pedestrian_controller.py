from controller import Supervisor
import random

robot    = Supervisor()
timestep = int(robot.getBasicTimeStep())

SPEED   = 0.4    # m/s — calm walking pace
FIXED_Z = 1.29   # vertical axis in this world — never touched

# Durations expressed in simulation steps (timestep = 8 ms → 125 steps/s)
INITIAL_STAND_STEPS = 1500  # ~12 s — stand still at simulation start before moving
MIN_WALK_STEPS = 750    # ~6 s  — long enough for drone to lock on and follow
MAX_WALK_STEPS = 2000   # ~16 s — pedestrian keeps going so drone has time to stabilise
MIN_WAIT_STEPS = 100    # ~0.8 s
MAX_WAIT_STEPS = 375    # ~3 s

# Cardinal directions: (dX, dY) — X=forward/backward, Y=left/right
DIRECTIONS = [
    ( 1,  0),   # forward
    (-1,  0),   # backward
    ( 0,  1),   # left
    ( 0, -1),   # right
]

trans_field = robot.getSelf().getField("translation")

initial_stand_remaining = INITIAL_STAND_STEPS
current_dir     = random.choice(DIRECTIONS)
steps_remaining = random.randint(MIN_WALK_STEPS, MAX_WALK_STEPS)
is_waiting      = False

print("[PED] Pedestrian started — standing still for initial period.")

while robot.step(timestep) != -1:
    dt = timestep / 1000.0

    # ── INITIAL STANDING PHASE ────────────────────────────────────────────────
    if initial_stand_remaining > 0:
        initial_stand_remaining -= 1
        if initial_stand_remaining == 0:
            print("[PED] Initial stand complete — starting movement.")
        continue

    # ── COUNT DOWN current action ─────────────────────────────────────────────
    steps_remaining -= 1

    if steps_remaining <= 0:
        if is_waiting:
            # Finished waiting → start walking in a new direction
            is_waiting      = False
            current_dir     = random.choice(DIRECTIONS)
            steps_remaining = random.randint(MIN_WALK_STEPS, MAX_WALK_STEPS)
        else:
            # Finished walking → 25 % chance to pause, otherwise new direction
            if random.random() < 0.25:
                is_waiting      = True
                steps_remaining = random.randint(MIN_WAIT_STEPS, MAX_WAIT_STEPS)
            else:
                current_dir     = random.choice(DIRECTIONS)
                steps_remaining = random.randint(MIN_WALK_STEPS, MAX_WALK_STEPS)

    # ── APPLY MOVEMENT ────────────────────────────────────────────────────────
    if is_waiting:
        continue

    pos  = trans_field.getSFVec3f()
    step = SPEED * dt

    trans_field.setSFVec3f([
        pos[0] + current_dir[0] * step,   # X  (forward / backward)
        pos[1] + current_dir[1] * step,   # Y  (left / right)
        FIXED_Z,                           # Z  always 1.29
    ])
