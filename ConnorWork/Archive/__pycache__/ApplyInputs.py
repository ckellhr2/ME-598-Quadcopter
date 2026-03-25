import pybullet as p

def apply_inputs(robot_id, omega_l, omega_r, max_torque=2.0):
    """
    Sends angular velocity commands to the left and right wheels
    of a differential-drive robot in PyBullet.
    """
    LEFT = 0
    RIGHT = 1

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