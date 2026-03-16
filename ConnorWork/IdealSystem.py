import numpy as np
import pybullet as p
import pybullet_data
import time
import os

# ==========================
# Quadrotor parameters
# ==========================
m = 0.5          # must match URDF total mass
g = 9.81
L = 0.15
Ix, Iy, Iz = 0.0023, 0.0023, 0.004
J = np.diag([Ix, Iy, Iz])
k_yaw = 0.01

NX = 12
NU = 4

# ==========================
# State from PyBullet
# ==========================
def get_state_from_bullet(body_id):
    pos, quat = p.getBasePositionAndOrientation(body_id)
    vel, ang_vel = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(quat)

    return np.array([
        pos[0], pos[1], pos[2],
        vel[0], vel[1], vel[2],
        roll, pitch, yaw,
        ang_vel[0], ang_vel[1], ang_vel[2]
    ])

# ==========================
# Apply rotor forces
# ==========================
def apply_motor_forces(body_id, u):
    f1, f2, f3, f4 = u

    # rotor positions in body frame
    rotor_positions = [
        [ L, 0, 0],   # front
        [ 0, L, 0],   # left
        [-L, 0, 0],   # back
        [ 0,-L, 0]    # right
    ]
    forces = [f1, f2, f3, f4]

    base_pos, base_quat = p.getBasePositionAndOrientation(body_id)
    R = np.array(p.getMatrixFromQuaternion(base_quat)).reshape(3, 3)

    for r_body, f in zip(rotor_positions, forces):
        # thrust along +Z body, rotated into world
        force_world = R @ np.array([0, 0, f])

        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=force_world.tolist(),
            posObj=r_body,
            flags=p.WORLD_FRAME
        )

# ==========================
# Cost
# ==========================
def make_cost_matrices():
    Q = np.diag([
        1.0, 1.0, 2.0,      # x, y, z
        100.0, 100.0, 30.0, # roll, pitch, yaw
        0.5, 0.5, 1.0,      # vx, vy, vz
        5.0, 5.0, 3.0       # p, q, r
        ])

    R_cost = np.diag([
        0.5, 0.5, 0.5, 0.5  # four motors / thrusts
        ])

    Qf = 10 * Q
    return Q, R_cost, Qf

def cost_stage(x, u, x_goal, Q, R_cost):
    dx = x - x_goal
    return dx.T @ Q @ dx + u.T @ R_cost @ u

def cost_final(x, x_goal, Qf):
    dx = x - x_goal
    return dx.T @ Qf @ dx

# ==========================
# iLQR (short horizon, linearized)
# ==========================
def ilqr(x0, x_goal, N, dt, Q, R_cost, Qf, u_init=None, max_iter=10):
    if u_init is None:
        u_hover = (m * g / 4.0) * np.ones(NU)
        u_seq = np.tile(u_hover, (N, 1))
    else:
        u_seq = u_init.copy()

    # simple linear model around hover
    A = np.eye(NX)
    A[0,3] = dt
    A[1,4] = dt
    A[2,5] = dt
    A[6,9] = dt
    A[7,10] = dt
    A[8,11] = dt

    B = np.zeros((NX, NU))
    B[5,:] = dt/m
    B[9,:]  = dt * np.array([0,  L,  0, -L])
    B[10,:] = dt * np.array([-L, 0,  L,  0])
    B[11,:] = dt * np.array([1, -1, 1, -1]) * k_yaw

    for _ in range(max_iter):
        x_seq = np.zeros((N+1, NX))
        x_seq[0] = x0
        cost_total = 0.0

        for k in range(N):
            x_seq[k+1] = A @ x_seq[k] + B @ u_seq[k]
            cost_total += cost_stage(x_seq[k], u_seq[k], x_goal, Q, R_cost)
        cost_total += cost_final(x_seq[-1], x_goal, Qf)

        Vx = 2 * Qf @ (x_seq[-1] - x_goal)
        Vxx = 2 * Qf

        K = np.zeros((N, NU, NX))
        k_ff = np.zeros((N, NU))

        for k in reversed(range(N)):
            dx = x_seq[k] - x_goal
            lx = 2 * Q @ dx
            lu = 2 * R_cost @ u_seq[k]
            lxx = 2 * Q
            luu = 2 * R_cost
            lux = np.zeros((NU, NX))

            Qx  = lx + A.T @ Vx
            Qu  = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_inv = np.linalg.inv(Quu + 1e-6*np.eye(NU))

            K[k]    = -Quu_inv @ Qux
            k_ff[k] = -Quu_inv @ Qu

            Vx  = Qx  + K[k].T @ Quu @ k_ff[k] + K[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k]    + K[k].T @ Qux + Qux.T @ K[k]

        alpha = 0.5
        for k in range(N):
            du = alpha * k_ff[k] + K[k] @ (x0 - x_seq[k])
            u_seq[k] += du
            hover = m * g / 4.0
            u_seq[k] = np.clip(u_seq[k], 0.0, hover * 1.5)

    return u_seq

# ==========================
# Main
# ==========================
def main():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -g)
    p.loadURDF("plane.urdf")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(script_dir, "quadrotor", "quadrotor.urdf")
    quad_id = p.loadURDF(urdf_path, [0, 0, 1])

    dt = 1/240
    N_horizon = 10
    replanning_steps = 5

    Q, R_cost, Qf = make_cost_matrices()

    x_goal = np.zeros(NX)
    x_goal[0:3] = [5.0, 5.0, 3.0]

    u_hover = np.ones(4) * (m * g / 4.0)
    u_seq = np.tile(u_hover, (N_horizon, 1))

    step = 0
    while True:
        x = get_state_from_bullet(quad_id)

        if step % replanning_steps == 0:
            u_seq = ilqr(x, x_goal, N_horizon, dt, Q, R_cost, Qf, u_init=u_seq)

        u = u_seq[0]
        apply_motor_forces(quad_id, u)
        u_seq = np.vstack([u_seq[1:], u_seq[-1]])

        p.stepSimulation()
        step += 1

        if np.linalg.norm(x[0:3] - x_goal[0:3]) < 0.2 and step > 300:
            print("Reached goal region.")
            break

    while True:
        p.stepSimulation()
        time.sleep(dt)

if __name__ == "__main__":
    main()