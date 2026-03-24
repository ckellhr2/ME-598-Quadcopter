import pybullet as p
import os
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
from DronePlots import plot_logs
from GoalChecker import reached_goal
from ApplyInputsNoisy import apply_inputs_noisy
from iLQRController import ilqr_plan


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

    goal = [7.0, 7.0]

    # --- Create marker for visualization
    goal_radius = 0.05
    goal_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=goal_radius,
        rgbaColor=[0, 1, 0, 1]   # green
    )
    goal_marker = p.createMultiBody(
        baseMass=0,              # no physics
        baseVisualShapeIndex=goal_visual,
        basePosition=[goal[0], goal[1], 0]  # sits on ground
    )
    #place obstacles, 
    obstacle_radius = .75
    obstacle1_position = [2,2]
    obstacle2_position = [3,4]
    obstacle3_position = [2,4]
    obstacle_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=obstacle_radius,
        rgbaColor=[1, 0, 0, 1]   # red
    )
    obstacle_1 = p.createMultiBody(
        baseMass=0,              # no physics
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[obstacle1_position[0],obstacle1_position[1], 0]  # 1/3 of the way to goal
    )
    obstacle_2 = p.createMultiBody(
        baseMass=0,              # no physics
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[obstacle2_position[0], obstacle2_position[1], 0]  
    )

    obstacle_3 = p.createMultiBody(
        baseMass=0,              # no physics
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[obstacle3_position[0],obstacle3_position[1], 0] 
    )

    # get initial state from PyBullet
    pos, orn = p.getBasePositionAndOrientation(robotId)
    x0, y0, _ = pos
    yaw0 = p.getEulerFromQuaternion(orn)[2]
    start_state = np.array([x0, y0, yaw0])

    # obstacles: (x, y, radius)
    obstacles = [
        (obstacle1_position[0], obstacle1_position[1], obstacle_radius),
        (obstacle2_position[0], obstacle2_position[1], obstacle_radius),
        (obstacle3_position[0], obstacle3_position[1], obstacle_radius),
    ]

    X_ref, U_ref = ilqr_plan(start_state, goal, obstacles)
    step_idx = 0

    # PyBullet runs at 240 Hz, iLQR at dt = 0.05 → 20 Hz
    sim_dt = 1/240
    ilqr_dt = 0.05
    ilqr_dt_steps = int(ilqr_dt / sim_dt)   # = 12

    ilqr_step_counter = 0

    for i in range(10000):

        # Stop if we reached the end of the iLQR plan
        if step_idx >= len(U_ref):
            print("iLQR horizon finished")
            break

        # Only advance iLQR control every 12 PyBullet steps
        if ilqr_step_counter == 0:
            v, w = U_ref[step_idx]
            step_idx += 1

            # Convert (v, w) → wheel speeds
            r = 0.05
            L = 0.20
            omega_r = (v + (L/2)*w) / r
            omega_l = (v - (L/2)*w) / r

        # Increment counter (wrap around every 12 steps)
        ilqr_step_counter = (ilqr_step_counter + 1) % ilqr_dt_steps

        # Apply wheel speeds
        apply_inputs_noisy(robotId, omega_l=omega_l, omega_r=omega_r) #update here to add noise
        log_state(robotId, omega_l, omega_r)

        p.stepSimulation()
        time.sleep(sim_dt)


        pos, orn = p.getBasePositionAndOrientation(robotId)
        x_bot, y_bot, z_bot = pos

        # Recenter camera on the robot
        p.resetDebugVisualizerCamera(
            cameraDistance=4.0,      # zoom level
            cameraYaw=45,            # angle around the robot
            cameraPitch=-30,         # tilt downward
            cameraTargetPosition=[x_bot, y_bot, 0.1]
        )

    p.disconnect()

    # show_plots=True to show, False to suppress
    plot_logs(log, show_plots=False)

if __name__ == "__main__":
    main()