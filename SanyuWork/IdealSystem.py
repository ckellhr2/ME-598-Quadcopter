import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import traceback

# ==========================
# Quadrotor parameters
# ==========================
g = 9.81
L = 0.15
k_yaw = 0.01
ILQR_REG = 1e-6

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

def get_total_mass(body_id):
    total_mass = 0.0
    num_joints = p.getNumJoints(body_id)

    for link_index in range(-1, num_joints):
        total_mass += p.getDynamicsInfo(body_id, link_index)[0]

    return total_mass

def keep_gui_alive(dt=1 / 240):
    while p.isConnected():
        p.stepSimulation()
        time.sleep(dt)

def compute_trajectory_cost(x_history, u_history, x_goal, Q, R, Qf):
    if not x_history:
        return 0.0

    total_cost = 0.0

    for x, u in zip(x_history[:-1], u_history):
        total_cost += cost_stage(x, u, x_goal, Q, R)

    total_cost += cost_final(x_history[-1], x_goal, Qf)
    return float(total_cost)

def run_simulation(
    start_pos=None,
    x_goal_pos=None,
    connection_mode=p.GUI,
    keep_alive=False,
    max_steps=20 * 240,
    log_interval=240,
    verbose=True,
    reuse_existing_connection=False,
):
    if start_pos is None:
        start_pos = [0, 0, 1]
    if x_goal_pos is None:
        x_goal_pos = [1.3, 2.1, 1.4]

    physics_client = None
    if reuse_existing_connection:
        if not p.isConnected():
            raise RuntimeError("reuse_existing_connection=True but no PyBullet connection is active.")
    else:
        physics_client = p.connect(connection_mode)

    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -g)
        p.loadURDF("plane.urdf")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(script_dir, "quadrotor", "quadrotor.urdf")
        quad_id = p.loadURDF(urdf_path, start_pos)

        m = get_total_mass(quad_id)
        inertia_diag = np.array(p.getDynamicsInfo(quad_id, -1)[2])
        if verbose:
            print("Total mass:", m)
            print("Base inertia:", inertia_diag)

        dt = 1 / 240
        N = 180
        replanning_steps = 10

        Q, R, Qf = make_cost_matrices()

        x_goal = np.zeros(NX)
        x_goal[0:3] = x_goal_pos

        u_hover = np.ones(4) * (m * g / 4.0)
        # Slight upward bias to encourage takeoff without making the climb too aggressive.
        u_seq = np.tile(u_hover * 1.01, (N, 1))

        step = 0
        x_history = []
        u_history = []
        if verbose:
            print("Simulation started.")

        while step < max_steps:
            x = get_state_from_bullet(quad_id)
            x_history.append(x.copy())

            if step % replanning_steps == 0:
                u_seq = ilqr(x, x_goal, N, dt, m, inertia_diag, Q, R, Qf, u_seq)

            u = u_seq[0]
            u_history.append(u.copy())
            apply_motor_forces(quad_id, u)
            u_seq = np.vstack([u_seq[1:], u_seq[-1]])

            p.stepSimulation()
            if connection_mode == p.GUI:
                time.sleep(dt)
            step += 1

            if verbose and log_interval and step % log_interval == 0:
                print(f"t={step * dt:.1f}s pos={x[0:3]} vel={x[3:6]} thrust={u}")

            pos_error = np.linalg.norm(x[0:3] - x_goal[0:3])
            vertical_speed = abs(x[5])

            if pos_error < 0.1 and vertical_speed < 0.15 and step > 300:
                if verbose:
                    print("Reached goal!")
                break

        if verbose and step >= max_steps:
            print("Max simulation time reached.")

        final_state = x_history[-1] if x_history else np.zeros(NX)
        trajectory_total_cost = compute_trajectory_cost(x_history, u_history, x_goal, Q, R, Qf)

        result = {
            "start_x": float(start_pos[0]),
            "start_y": float(start_pos[1]),
            "start_z": float(start_pos[2]),
            "goal_x": float(x_goal_pos[0]),
            "goal_y": float(x_goal_pos[1]),
            "goal_z": float(x_goal_pos[2]),
            "steps": int(step),
            "reached_goal": bool(
                np.linalg.norm(final_state[0:3] - x_goal[0:3]) < 0.1 and abs(final_state[5]) < 0.15
            ),
            "final_x": float(final_state[0]),
            "final_y": float(final_state[1]),
            "final_z": float(final_state[2]),
            "final_vx": float(final_state[3]),
            "final_vy": float(final_state[4]),
            "final_vz": float(final_state[5]),
            "final_position_error": float(np.linalg.norm(final_state[0:3] - x_goal[0:3])),
            "trajectory_total_cost": trajectory_total_cost,
        }

        if keep_alive and connection_mode == p.GUI:
            print(f"Total trajectory cost: {trajectory_total_cost:.6f}")
            keep_gui_alive(dt)

        return result

    except Exception:
        traceback.print_exc()
        if keep_alive and connection_mode == p.GUI:
            print("Simulation failed. Keeping the PyBullet window open.")
            keep_gui_alive()
        raise
    finally:
        if physics_client is not None and p.isConnected(physics_client):
            p.disconnect(physics_client)

# ==========================
# Apply rotor forces
# ==========================
def apply_motor_forces(body_id, u):
    rotor_positions_body = np.array([
        [L, 0, 0],
        [0, L, 0],
        [-L, 0, 0],
        [0, -L, 0]
    ])
    yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])

    for r_body, f in zip(rotor_positions_body, u):
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=[0.0, 0.0, float(f)],
            posObj=r_body.tolist(),
            flags=p.LINK_FRAME
        )

    yaw_torque = float(k_yaw * np.dot(yaw_signs, u))
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=[0.0, 0.0, yaw_torque],
        flags=p.LINK_FRAME
    )

# ==========================
# Cost
# ==========================
def make_cost_matrices():
    Q = np.diag([
        8.0, 8.0, 80.0,      # position
        1.0, 1.0, 25.0,      # velocity
        50.0, 50.0, 20.0,    # angles
        5.0, 5.0, 8.0        # angular rates
    ])

    R_cost = np.diag([0.05, 0.05, 0.05, 0.05])
    Qf = 30 * Q

    return Q, R_cost, Qf

def cost_stage(x, u, x_goal, Q, R):
    dx = x - x_goal
    return dx.T @ Q @ dx + u.T @ R @ u

def cost_final(x, x_goal, Qf):
    dx = x - x_goal
    return dx.T @ Qf @ dx

# ==========================
# iLQR
# ==========================
def build_linearized_dynamics(dt, m, inertia_diag, yaw):
    ix, iy, iz = inertia_diag
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    A = np.eye(NX)
    A[0, 3] = dt
    A[1, 4] = dt
    A[2, 5] = dt
    A[6, 9] = dt
    A[7, 10] = dt
    A[8, 11] = dt

    # Small-angle translational coupling around the current yaw.
    A[3, 6] = -g * s_yaw * dt
    A[3, 7] =  g * c_yaw * dt
    A[4, 6] = -g * c_yaw * dt
    A[4, 7] = -g * s_yaw * dt

    B = np.zeros((NX, NU))
    B[5, :] = dt / m
    B[9, :] = dt * np.array([0.0, L / ix, 0.0, -L / ix])
    B[10, :] = dt * np.array([-L / iy, 0.0, L / iy, 0.0])
    B[11, :] = dt * k_yaw * np.array([1.0, -1.0, 1.0, -1.0]) / iz

    return A, B

def rollout_dynamics(x0, delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover):
    x_seq = np.zeros((len(delta_u_seq) + 1, NX))
    u_abs_seq = np.zeros((len(delta_u_seq), NU))
    x_seq[0] = x0
    total_cost = 0.0

    for k, delta_u in enumerate(delta_u_seq):
        A, B = build_linearized_dynamics(dt, m, inertia_diag, x_seq[k, 8])
        u_abs = np.clip(hover + delta_u, 0.0, 2.5 * hover)
        delta_u_clipped = u_abs - hover
        x_seq[k + 1] = A @ x_seq[k] + B @ delta_u_clipped
        u_abs_seq[k] = u_abs
        total_cost += cost_stage(x_seq[k], delta_u_clipped, x_goal, Q, R)

    total_cost += cost_final(x_seq[-1], x_goal, Qf)
    return x_seq, u_abs_seq, total_cost

def ilqr(x0, x_goal, N, dt, m, inertia_diag, Q, R, Qf, u_init=None, max_iter=25):
    hover = m * g / 4.0

    if u_init is None:
        delta_u_seq = np.zeros((N, NU))
    else:
        delta_u_seq = np.clip(u_init - hover, -hover, 1.5 * hover)

    for _ in range(max_iter):
        x_seq, _, current_cost = rollout_dynamics(
            x0, delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover
        )

        Vx = 2 * Qf @ (x_seq[-1] - x_goal)
        Vxx = 2 * Qf

        K = np.zeros((N, NU, NX))
        k_ff = np.zeros((N, NU))

        for k in reversed(range(N)):
            A, B = build_linearized_dynamics(dt, m, inertia_diag, x_seq[k, 8])
            dx = x_seq[k] - x_goal

            lx = 2 * Q @ dx
            lu = 2 * R @ delta_u_seq[k]
            lxx = 2 * Q
            luu = 2 * R
            lux = np.zeros((NU, NX))

            Qx  = lx + A.T @ Vx
            Qu  = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_reg = Quu + ILQR_REG * np.eye(NU)
            Quu_inv = np.linalg.inv(Quu_reg)

            K[k] = -Quu_inv @ Qux
            k_ff[k] = -Quu_inv @ Qu

            Vx = Qx + K[k].T @ Quu @ k_ff[k] + K[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            Vxx = 0.5 * (Vxx + Vxx.T)

        improved = False
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
            trial_delta_u_seq = np.zeros_like(delta_u_seq)
            trial_x_seq = np.zeros_like(x_seq)
            trial_x_seq[0] = x0

            for k in range(N):
                dx = trial_x_seq[k] - x_seq[k]
                trial_delta_u_seq[k] = delta_u_seq[k] + alpha * k_ff[k] + K[k] @ dx
                A, B = build_linearized_dynamics(dt, m, inertia_diag, trial_x_seq[k, 8])
                u_abs = np.clip(hover + trial_delta_u_seq[k], 0.0, 2.5 * hover)
                trial_delta_u_seq[k] = u_abs - hover
                trial_x_seq[k + 1] = A @ trial_x_seq[k] + B @ trial_delta_u_seq[k]

            _, trial_u_abs_seq, trial_cost = rollout_dynamics(
                x0, trial_delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover
            )

            if trial_cost < current_cost:
                delta_u_seq = trial_u_abs_seq - hover
                improved = True
                break

        if not improved:
            break

    return np.clip(hover + delta_u_seq, 0.0, 2.5 * hover)

# ==========================
# Main
# ==========================
def main():
    result = run_simulation(
        start_pos=[0, 0, 1],
        x_goal_pos=[1.3, 2.1, 1.4],
        connection_mode=p.GUI,
        keep_alive=True,
    )
    print(f"Total trajectory cost: {result['trajectory_total_cost']:.6f}")

# ==========================
if __name__ == "__main__":
    main()
