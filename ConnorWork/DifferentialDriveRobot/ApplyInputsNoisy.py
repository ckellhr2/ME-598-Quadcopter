import pybullet as p
import numpy as np

def apply_inputs_noisy(
    robot_id,
    omega_l,
    omega_r,
    max_torque=2.0,
    force_std=0.01,
    torque_std=0.05
):
    """
    Differential-drive wheel control + Gaussian noise disturbances.
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

    # --- Gaussian noise in robot frame ---
    fx_r = np.random.normal(0, force_std)
    fy_r = np.random.normal(0, force_std)
    tz_r = np.random.normal(0, torque_std)

    # Convert robot-frame force to world frame
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    rot_mat = p.getMatrixFromQuaternion(orn)
    R = np.array(rot_mat).reshape(3, 3)

    # Robot-frame force vector
    f_robot = np.array([fx_r, fy_r, 0.0])
    f_world = R @ f_robot

    # Apply force at COM
    p.applyExternalForce(
        objectUniqueId=robot_id,
        linkIndex=-1,
        forceObj=f_world.tolist(),
        posObj=[0, 0, 0],
        flags=p.WORLD_FRAME
    )

    t_robot = np.array([0.0, 0.0, tz_r])
    t_world = R @ t_robot

    p.applyExternalTorque(
        objectUniqueId=robot_id,
        linkIndex=-1,
        torqueObj=t_world.tolist(),
        flags=p.WORLD_FRAME
    )