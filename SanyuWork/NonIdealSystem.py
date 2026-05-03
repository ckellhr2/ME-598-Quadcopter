import os
import time

import numpy as np
import pybullet as p
import pybullet_data

from IdealSystem import (
    NX,
    g,
    L,
    k_yaw,
    get_state_from_bullet,
    get_total_mass,
    make_cost_matrices,
    ilqr,
)


def apply_motor_forces_scaled(body_id, u, plant_thrust_scale=1.0, plant_yaw_coeff_scale=1.0):
    rotor_positions_body = np.array([
        [L, 0, 0],
        [0, L, 0],
        [-L, 0, 0],
        [0, -L, 0],
    ])
    yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])

    scaled_u = np.asarray(u, dtype=float) * float(plant_thrust_scale)

    for r_body, f in zip(rotor_positions_body, scaled_u):
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=[0.0, 0.0, float(f)],
            posObj=r_body.tolist(),
            flags=p.LINK_FRAME,
        )

    yaw_torque = float(k_yaw * plant_yaw_coeff_scale * np.dot(yaw_signs, scaled_u))
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=[0.0, 0.0, yaw_torque],
        flags=p.LINK_FRAME,
    )


def run_simulation(
    start_pos=None,
    x_goal_pos=None,
    connection_mode=p.GUI,
    keep_alive=False,
    max_steps=12 * 240,
    log_interval=240,
    verbose=True,
    reuse_existing_connection=False,
    sample_period_sec=0.1,
    plant_mass_scale=1.0,
    plant_inertia_scale=1.0,
    plant_thrust_scale=1.0,
    plant_yaw_coeff_scale=1.0,
):
    if start_pos is None:
        start_pos = [0.0, 0.0, 1.0]
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

        m_nominal = get_total_mass(quad_id)
        inertia_nominal = np.array(p.getDynamicsInfo(quad_id, -1)[2], dtype=float)

        p.changeDynamics(
            quad_id,
            -1,
            mass=float(m_nominal * plant_mass_scale),
            localInertiaDiagonal=(inertia_nominal * plant_inertia_scale).tolist(),
        )

        dt = 1 / 240
        N = 180
        replanning_steps = 10
        sample_stride = max(1, int(round(sample_period_sec / dt)))

        Q, R, Qf = make_cost_matrices()
        x_goal = np.zeros(NX)
        x_goal[0:3] = x_goal_pos

        # Controller remains nominal / ideal.
        u_hover = np.ones(4) * (m_nominal * g / 4.0)
        u_seq = np.tile(u_hover * 1.01, (N, 1))

        trajectory_rows = []
        step = 0
        if verbose:
            print("Fixed epistemic uncertainty simulation started.")
            print(
                f"Plant scales: mass={plant_mass_scale}, inertia={plant_inertia_scale}, "
                f"thrust={plant_thrust_scale}, yaw_coeff={plant_yaw_coeff_scale}"
            )

        while step < max_steps:
            time_sec = step * dt
            x = get_state_from_bullet(quad_id)

            if step % sample_stride == 0:
                trajectory_rows.append({
                    "time_step": int(step),
                    "time_sec": float(time_sec),
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "z": float(x[2]),
                    "effective_mass_scale": float(plant_mass_scale),
                    "effective_inertia_scale": float(plant_inertia_scale),
                    "effective_thrust_scale": float(plant_thrust_scale),
                    "effective_yaw_coeff_scale": float(plant_yaw_coeff_scale),
                })

            if step % replanning_steps == 0:
                u_seq = ilqr(x, x_goal, N, dt, m_nominal, inertia_nominal, Q, R, Qf, u_seq)

            u = u_seq[0]
            apply_motor_forces_scaled(
                quad_id,
                u,
                plant_thrust_scale=plant_thrust_scale,
                plant_yaw_coeff_scale=plant_yaw_coeff_scale,
            )
            u_seq = np.vstack([u_seq[1:], u_seq[-1]])

            p.stepSimulation()
            if connection_mode == p.GUI:
                time.sleep(dt)
            step += 1

            if verbose and log_interval and step % log_interval == 0:
                print(f"t={step * dt:.1f}s pos={x[0:3]}")

            pos_error = np.linalg.norm(x[0:3] - x_goal[0:3])
            vertical_speed = abs(x[5])
            if pos_error < 0.1 and vertical_speed < 0.15 and step > 300:
                if verbose:
                    print("Reached goal.")
                break

        final_state = get_state_from_bullet(quad_id)
        return {
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
            "trajectory_rows": trajectory_rows,
        }

    finally:
        if physics_client is not None and p.isConnected(physics_client):
            p.disconnect(physics_client)
