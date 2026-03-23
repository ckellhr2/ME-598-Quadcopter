import numpy as np
import pybullet as p
import math
from scipy.linalg import solve_continuous_are

def lqr(A, B, Q, R):
    X = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ X)
    return K

def goal_controller_lqr(body_id, goal):
    # Pose
    pos, orn = p.getBasePositionAndOrientation(body_id)
    x, y, _ = pos
    yaw = p.getEulerFromQuaternion(orn)[2]

    gx, gy = goal

    # Errors
    dx = gx - x
    dy = gy - y
    distance = math.sqrt(dx*dx + dy*dy)

    # Forward velocity reference (critical!)
    v_ref = min(1.0, distance)

    # Lateral error in robot frame
    e_y = dx * math.sin(yaw) - dy * math.cos(yaw)

    # Heading error
    desired_yaw = math.atan2(dy, dx)
    e_theta = desired_yaw - yaw
    e_theta = (e_theta + math.pi) % (2 * math.pi) - math.pi

    # State vector
    e = np.array([[e_y],
                  [e_theta]])

    # Linearized system
    A = np.array([[0.0,      v_ref],
                  [0.0,      0.0  ]])

    B = np.array([[0.0],
                  [1.0]])

    # Costs
    Q = np.diag([6.0, 3.0])
    R = np.array([[0.4]])

    # LQR gain
    K = lqr(A, B, Q, R)

    # Control law: w = -K e
    w = float(-(K @ e)[0])

    # Convert (v_ref, w) → wheel speeds
    r = 0.05
    L = 0.20

    omega_r = (v_ref + (L/2)*w) / r
    omega_l = (v_ref - (L/2)*w) / r

    return omega_r, omega_l