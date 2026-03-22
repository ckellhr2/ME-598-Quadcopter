import pybullet as p
import os
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
from DronePlots import plot_logs
from GoalChecker import reached_goal
from ApplyInputs import apply_inputs
from LQRController import goal_controller_lqr

log = []  # log for states + inputs
tolerance=0.2 #tolerance for goal checking to see if position has been reached


def log_state(body_id, omega_l, omega_r):
    pos, orn = p.getBasePositionAndOrientation(body_id)
    lin_vel, ang_vel = p.getBaseVelocity(body_id)
    roll, pitch, yaw = p.getEulerFromQuaternion(orn)

    entry = {
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "vx": lin_vel[0],
        "vy": lin_vel[1],
        "vz": lin_vel[2],
        "wx": ang_vel[0],
        "wy": ang_vel[1],
        "wz": ang_vel[2],
        "omega_l": omega_l,
        "omega_r": omega_r,
    }

    log.append(entry)

def main():
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = os.path.join(os.path.dirname(__file__), "diff_drive.urdf")
    robotId = p.loadURDF(urdf_path, startPos, startOrientation)

    goal = [2.0, 2.0]

    for i in range(10000):
        if reached_goal(robotId, goal,tolerance):
            print("Goal reached!")
            break
        
        omega_r, omega_l = goal_controller_lqr(robotId, goal)
        apply_inputs(robotId, omega_l=omega_l, omega_r=omega_r)
        log_state(robotId, omega_l, omega_r)

        p.stepSimulation()
        time.sleep(1. / 240.)

    p.disconnect()

    # show_plots=True to show, False to suppress
    plot_logs(log, show_plots=False)


if __name__ == "__main__":
    main()