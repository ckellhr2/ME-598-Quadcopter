import pybullet as p
import numpy as np

def apply_inputs(
    robot_id,
    omega_l,
    omega_r,
    max_torque=5.0,
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

