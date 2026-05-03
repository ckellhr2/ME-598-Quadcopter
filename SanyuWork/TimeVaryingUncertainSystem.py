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


def apply_motor_forces_scaled(body_id, u, thrust_scale):
    rotor_positions_body = np.array([
        [L, 0, 0],
        [0, L, 0],
        [-L, 0, 0],
        [0, -L, 0],
    ])
    yaw_signs = np.array([1.0, -1.0, 1.0, -1.0])

    scaled_u = thrust_scale * np.asarray(u, dtype=float)

    for r_body, f in zip(rotor_positions_body, scaled_u):
        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=-1,
            forceObj=[0.0, 0.0, float(f)],
            posObj=r_body.tolist(),
            flags=p.LINK_FRAME,
        )

    yaw_torque = float(k_yaw * np.dot(yaw_signs, scaled_u))
    p.applyExternalTorque(
        objectUniqueId=body_id,
        linkIndex=-1,
        torqueObj=[0.0, 0.0, yaw_torque],
        flags=p.LINK_FRAME,
    )


def smoothstep01(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def evaluate_time_varying_profile(profile, time_sec):
    thrust_drop_start_sec = float(profile["thrust_drop_start_sec"])
    thrust_drop_end_sec = float(profile["thrust_drop_end_sec"])
    thrust_drop_fraction = float(profile["thrust_drop_fraction"])

    if thrust_drop_end_sec <= thrust_drop_start_sec:
        thrust_progress = 1.0 if time_sec >= thrust_drop_end_sec else 0.0
    else:
        thrust_progress = smoothstep01(
            (time_sec - thrust_drop_start_sec) / (thrust_drop_end_sec - thrust_drop_start_sec)
        )
    thrust_scale = 1.0 - thrust_drop_fraction * thrust_progress

    wind_fx = float(
        profile["wind_bias_x"]
        + profile["wind_amp_x"] * np.sin(2.0 * np.pi * profile["wind_freq_x_hz"] * time_sec + profile["wind_phase_x"])
    )
    wind_fy = float(
        profile["wind_bias_y"]
        + profile["wind_amp_y"] * np.sin(2.0 * np.pi * profile["wind_freq_y_hz"] * time_sec + profile["wind_phase_y"])
    )

    gust_start_sec = float(profile["gust_start_sec"])
    gust_duration_sec = float(profile["gust_duration_sec"])
    gust_stop_sec = gust_start_sec + gust_duration_sec
    if gust_start_sec <= time_sec <= gust_stop_sec and gust_duration_sec > 0.0:
        gust_phase = (time_sec - gust_start_sec) / gust_duration_sec
        gust_envelope = np.sin(np.pi * gust_phase)
        wind_fx += float(profile["gust_force_x"]) * gust_envelope
        wind_fy += float(profile["gust_force_y"]) * gust_envelope
        wind_fz = float(profile["gust_force_z"]) * gust_envelope
    else:
        wind_fz = 0.0

    return {
        "thrust_scale": float(thrust_scale),
        "wind_force_world": np.array([wind_fx, wind_fy, wind_fz], dtype=float),
    }


def run_simulation(
    start_pos=None,
    x_goal_pos=None,
    disturbance_profile=None,
    connection_mode=p.GUI,
    keep_alive=False,
    max_steps=20 * 240,
    log_interval=240,
    verbose=True,
    reuse_existing_connection=False,
    sample_period_sec=0.1,
):
    if start_pos is None:
        start_pos = [0.0, 0.0, 1.0]
    if x_goal_pos is None:
        x_goal_pos = [1.3, 2.1, 1.4]
    if disturbance_profile is None:
        raise ValueError("disturbance_profile must be provided for time-varying uncertainty runs.")

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
        inertia_diag = np.array(p.getDynamicsInfo(quad_id, -1)[2], dtype=float)
        dt = 1 / 240
        N = 180
        replanning_steps = 10
        sample_stride = max(1, int(round(sample_period_sec / dt)))

        Q, R, Qf = make_cost_matrices()
        x_goal = np.zeros(NX)
        x_goal[0:3] = x_goal_pos

        u_hover = np.ones(4) * (m * g / 4.0)
        u_seq = np.tile(u_hover * 1.01, (N, 1))

        trajectory_rows = []
        step = 0
        if verbose:
            print("Time-varying uncertainty simulation started.")
            print(f"Disturbance profile: {disturbance_profile}")

        while step < max_steps:
            time_sec = step * dt
            x = get_state_from_bullet(quad_id)

            profile_state = evaluate_time_varying_profile(disturbance_profile, time_sec)
            thrust_scale = profile_state["thrust_scale"]
            wind_force_world = profile_state["wind_force_world"]

            if step % sample_stride == 0:
                trajectory_rows.append({
                    "time_step": int(step),
                    "time_sec": float(time_sec),
                    "x": float(x[0]),
                    "y": float(x[1]),
                    "z": float(x[2]),
                    "effective_thrust_scale": float(thrust_scale),
                    "wind_fx": float(wind_force_world[0]),
                    "wind_fy": float(wind_force_world[1]),
                    "wind_fz": float(wind_force_world[2]),
                })

            if step % replanning_steps == 0:
                u_seq = ilqr(x, x_goal, N, dt, m, inertia_diag, Q, R, Qf, u_seq)

            u = u_seq[0]
            apply_motor_forces_scaled(quad_id, u, thrust_scale)
            p.applyExternalForce(
                objectUniqueId=quad_id,
                linkIndex=-1,
                forceObj=wind_force_world.tolist(),
                posObj=[0.0, 0.0, 0.0],
                flags=p.WORLD_FRAME,
            )
            u_seq = np.vstack([u_seq[1:], u_seq[-1]])

            p.stepSimulation()
            if connection_mode == p.GUI:
                time.sleep(dt)
            step += 1

            if verbose and log_interval and step % log_interval == 0:
                print(
                    f"t={step * dt:.1f}s pos={x[0:3]} "
                    f"thrust_scale={thrust_scale:.3f} wind={wind_force_world}"
                )

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
