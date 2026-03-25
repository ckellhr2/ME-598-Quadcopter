import pybullet as p
import numpy as np

def reached_goal(body_id, goal, tolerance):
    """
    Returns True if:
      1. Robot is within pos_tol of goal (x,y)
      2. Linear and angular velocity magnitudes are very small
    """

    # --- Position check ---
    pos, _ = p.getBasePositionAndOrientation(body_id)
    x, y, _ = pos
    gx, gy = goal

    pos_ok = abs(x - gx) < tolerance and abs(y - gy) < tolerance

    # --- Velocity check (zero tolerance) ---
    lin_vel, ang_vel = p.getBaseVelocity(body_id)

    lin_mag = np.linalg.norm(lin_vel)
    ang_mag = np.linalg.norm(ang_vel)

    vel_ok = (lin_mag < 1) and (ang_mag < 1)

    return pos_ok and vel_ok