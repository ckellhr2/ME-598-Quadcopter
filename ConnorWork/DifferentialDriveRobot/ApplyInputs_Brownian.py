import pybullet as p
import numpy as np

def apply_inputs_Brownian(
    robot_id,
    omega_l,
    omega_r,
    dt,
    max_torque=2.0,
    force_std=1,
    torque_std=5
):
    """
    Differential-drive wheel control + Brownian motion disturbances.
    Noise is applied in the ROBOT FRAME:
        - force in x, y
        - torque about z
    """
    LEFT = 0
    RIGHT = 1

    # --- Wheel velocity control ---
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=LEFT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=omega_l,
        force=max_torque
    )

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=RIGHT,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=omega_r,
        force=max_torque
    )

    # --- Brownian motion noise (scaled by sqrt(dt)) ---
    fx_r = np.random.normal(0, force_std * np.sqrt(dt))
    fy_r = np.random.normal(0, force_std * np.sqrt(dt))
    tz_r = np.random.normal(0, torque_std * np.sqrt(dt))

    # Convert robot-frame force to world frame
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    rot_mat = p.getMatrixFromQuaternion(orn)
    R = np.array(rot_mat).reshape(3, 3)

    f_world = R @ np.array([fx_r, fy_r, 0.0])
    t_world = R @ np.array([0.0, 0.0, tz_r])

    # Apply force at COM
    p.applyExternalForce(
        objectUniqueId=robot_id,
        linkIndex=-1,
        forceObj=f_world.tolist(),
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME
    )

    # Apply torque
    p.applyExternalTorque(
        objectUniqueId=robot_id,
        linkIndex=-1,
        torqueObj=t_world.tolist(),
        flags=p.WORLD_FRAME
    )