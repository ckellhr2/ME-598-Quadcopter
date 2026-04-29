import numpy as np
import pybullet as p
import pybullet_data
import matplotlib
import time
import os

# ==========================
# Quadrotor parameters
# ==========================
m = 0.5
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
        force_world = R @ np.array([0, 0, f])
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=force_world.tolist(),
            posObj=r_body,
            flags=p.WORLD_FRAME
        )

# ==========================
# Nonlinear dynamics f(x,u)
# ==========================
def quad_dynamics(x, u):
    px, py, pz = x[0:3]
    vx, vy, vz = x[3:6]
    roll, pitch, yaw = x[6:9]
    p_rate, q_rate, r_rate = x[9:12]

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
        [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
        [-sp,   sr*cp,            cr*cp]
    ])

    f1, f2, f3, f4 = u
    Fz = f1 + f2 + f3 + f4

    acc = (R @ np.array([0, 0, Fz])) / m - np.array([0, 0, g])

    tau = np.array([
        L * (f2 - f4),
        L * (f3 - f1),
        k_yaw * (f1 - f2 + f3 - f4)
    ])
    omega = np.array([p_rate, q_rate, r_rate])
    ang_acc = np.linalg.inv(J) @ (tau - np.cross(omega, J @ omega))

    euler_dot = np.zeros(3)
    euler_dot[0] = p_rate + q_rate*np.sin(roll)*np.tan(pitch) + r_rate*np.cos(roll)*np.tan(pitch)
    euler_dot[1] = q_rate*np.cos(roll) - r_rate*np.sin(roll)
    euler_dot[2] = q_rate*np.sin(roll)/np.cos(pitch) + r_rate*np.cos(roll)/np.cos(pitch)

    xdot = np.zeros(12)
    xdot[0:3] = x[3:6]
    xdot[3:6] = acc
    xdot[6:9] = euler_dot
    xdot[9:12] = ang_acc

    return xdot

# ==========================
# Linearization: A,B at (x,u)
# ==========================
def linearize_dynamics(x, u, dt):
    NX = len(x)
    NU = len(u)

    A = np.zeros((NX, NX))
    B = np.zeros((NX, NU))

    fx = quad_dynamics(x, u)
    eps = 1e-5

    for i in range(NX):
        dx = np.zeros(NX)
        dx[i] = eps
        A[:, i] = (quad_dynamics(x + dx, u) - fx) / eps

    for i in range(NU):
        du = np.zeros(NU)
        du[i] = eps
        B[:, i] = (quad_dynamics(x, u + du) - fx) / eps

    A_d = np.eye(NX) + A * dt
    B_d = B * dt

    return A_d, B_d

# ==========================
# Cost
# ==========================
def make_cost_matrices():
    Q = np.diag([
        1.0, 1.0, 100.0,     # x, y, z (z heavy)
        300.0, 300.0, 30.0, # roll, pitch, yaw (roll/pitch very heavy)
        1.0, 1.0, 20.0,     # vx, vy, vz (vz heavy)
        5.0, 5.0, 3.0       # p, q, r
    ])

    R_cost = np.diag([
        0.5, 0.5, 0.5, 0.5
    ])

    Qf = 10 * Q
    return Q, R_cost, Qf

def cost_stage(x, u, x_goal, Q, R_cost):
    dx = x - x_goal
    base_cost = dx.T @ Q @ dx + u.T @ R_cost @ u

    roll, pitch = x[6], x[7]
    tilt_cost = 50.0 * (roll**2 + pitch**2)

    u_hover = m * g / 4.0
    u_dev = u - u_hover
    hover_dev_cost = u_dev.T @ (0.1 * np.eye(4)) @ u_dev

    # NEW: penalize total thrust away from mg
    total_thrust = np.sum(u)
    thrust_err = total_thrust - m * g
    thrust_cost = 50.0 * thrust_err**2

    return base_cost + tilt_cost + hover_dev_cost + thrust_cost

def cost_final(x, x_goal, Qf):
    dx = x - x_goal
    return dx.T @ Qf @ dx

# ==========================
# iLQR with state-dependent linearization
# ==========================
def ilqr(x0, x_goal, N, dt, Q, R_cost, Qf, u_init=None, max_iter=10):
    if u_init is None:
        u_hover = (m * g / 4.0) * np.ones(NU)
        u_seq = np.tile(u_hover, (N, 1))
    else:
        u_seq = u_init.copy()

    for _ in range(max_iter):
        x_seq = np.zeros((N+1, NX))
        x_seq[0] = x0
        A_seq = np.zeros((N, NX, NX))
        B_seq = np.zeros((N, NX, NU))
        cost_total = 0.0

        for k in range(N):
            A_k, B_k = linearize_dynamics(x_seq[k], u_seq[k], dt)
            A_seq[k] = A_k
            B_seq[k] = B_k
            x_seq[k+1] = A_k @ x_seq[k] + B_k @ u_seq[k]
            cost_total += cost_stage(x_seq[k], u_seq[k], x_goal, Q, R_cost)
        cost_total += cost_final(x_seq[-1], x_goal, Qf)

        Vx = 2 * Qf @ (x_seq[-1] - x_goal)
        Vxx = 2 * Qf

        K = np.zeros((N, NU, NX))
        k_ff = np.zeros((N, NU))

        for k in reversed(range(N)):
            A = A_seq[k]
            B = B_seq[k]
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
            u_seq[k] = np.clip(u_seq[k], 0.7 * hover, 1.5 * hover)
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
    N_horizon = 40
    replanning_steps = 5

    Q, R_cost, Qf = make_cost_matrices()

    x_goal = np.zeros(NX)
    x_goal[0:3] = [5.0, 5.0, 1.0]

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

        if np.linalg.norm(x[0:3] - x_goal[0:3]) < 0.2 and step > 600:
            print("Reached goal region.")
            break

    while True:
        p.stepSimulation()
        time.sleep(dt)

if __name__ == "__main__":
    main()