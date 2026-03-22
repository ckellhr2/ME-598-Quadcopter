import numpy as np
import pybullet as p
import math
from scipy.linalg import solve_continuous_are

def lqr(A, B, Q, R):
    X = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ (B.T @ X)
    return K

def goal_controller_lqr(body_id, goal):
    # Robot pose
    pos, orn = p.getBasePositionAndOrientation(body_id)
    x, y, _ = pos
    yaw = p.getEulerFromQuaternion(orn)[2]

    gx, gy = goal

    # Error in world frame
    ex = gx - x
    ey = gy - y
    etheta = math.atan2(ey, ex) - yaw

    # Wrap angle
    etheta = (etheta + math.pi) % (2 * math.pi) - math.pi

    # State vector
    e = np.matrix([[ex], [ey], [etheta]])

    # Robot parameters, must match Diff_drive.urdf
    r = 0.05   # wheel radius
    L = 0.20   # wheel base

    # Actual State Space
    # A = np.zeros((3,3)) #robot only moves when wheels turn so states have no impact on dynamics
    # B = np.matrix([
    #   [-r/2*cos(theta), -r/2*cos(theta)],
    #   [-r/2*sin(theta), -r/2*sin(theta)],
    #   [r/L, -r/L]
    #])

    # Linearization around small error
    A = np.matrix([
        [0, 0, 0],
        [0, 0, -1],   # ey_dot ≈ -w, needed to make system fully controllable
        [0, 0, 0]
    ])

    # Map wheel speeds to (v, w)
    # v = r/2 (wr + wl)
    # w = r/L (wr - wl)
    B = np.matrix([
        [ r/2,     r/2     ],
        [ 0,       0       ],
        [ r/L,    -r/L     ]
    ])

    # Cost matrices
    Q = np.diag([3.0, 3.0, 2.0]) #error costs
    R = np.diag([0.1, 0.1]) #control costs

    # Compute LQR gain
    K = lqr(A, B, Q, R)

    # Control law: u = -K e
    u = -K @ e

    omega_r = float(u[0])
    omega_l = float(u[1])

    return omega_r, omega_l