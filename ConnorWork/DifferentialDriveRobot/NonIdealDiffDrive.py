import pybullet as p
import os
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
from DronePlots import plot_logs
from ApplyInputs_Brownian import apply_inputs_Brownian
from ideal_iLQRController import ilqr_plan

gui = False

log = {
    "x": [],
    "t": [],
}

def log_states_and_time(log, x, t,):
    # --- Log trajectory ---
    log["t"].append(t)
    log["x"].append(x.copy())



def main(startpos=[-1,0,0], goal = [4, 4],obstacle1_position = [2,2],
         obstacle2_position = [3,1],obstacle3_position = [1,3]):
    if gui:
        physicsClient = p.connect(p.GUI)
    else:
        physicsClient = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    planeId = p.loadURDF("plane.urdf")
    startOrientation = p.getQuaternionFromEuler([0, 0, 45])

    urdf_path = os.path.join(os.path.dirname(__file__), "diff_drive.urdf")
    robotId = p.loadURDF(urdf_path, startpos, startOrientation)

    

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
    obstacle_radius = .375
    obstacle_collision = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=obstacle_radius
    )
    obstacle_visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=obstacle_radius,
        rgbaColor=[1, 0, 0, 1]   # red
    )
    obstacle_1 = p.createMultiBody(
        baseMass=0,             
        baseCollisionShapeIndex=obstacle_collision,
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[obstacle1_position[0],obstacle1_position[1], 0] 
    )
    obstacle_2 = p.createMultiBody(
        baseMass=0,              
        baseCollisionShapeIndex=obstacle_collision,
        baseVisualShapeIndex=obstacle_visual,
        basePosition=[obstacle2_position[0], obstacle2_position[1], 0]  
    )

    obstacle_3 = p.createMultiBody(
        baseMass=0,             
        baseCollisionShapeIndex=obstacle_collision,
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

    # PyBullet runs at 240 Hz, iLQR at dt = 0.05 → 20 Hz
    sim_dt = 1/240
    ilqr_dt = 0.05
    ilqr_dt_steps = int(ilqr_dt / sim_dt)   # = 12

    ilqr_step_counter = 0
    step_idx = 0

    # --- initial plan ---
    X_ref, U_ref = ilqr_plan(start_state, goal, obstacles)
    terminal = False #used to tell logger that this step is the last if true for terminal cost
    
    for i in range(2400): #10 seconds
        # --- get current robot state ---
        pos, orn = p.getBasePositionAndOrientation(robotId)
        x_bot, y_bot, _ = pos
        yaw_bot = p.getEulerFromQuaternion(orn)[2]
        current_state = np.array([x_bot, y_bot, yaw_bot])

        need_replan = False #used to call planner again if true
        # 1. End of horizon
        if step_idx >= 70: #plan to rerun iLQR at 70 and 140 steps out of 200
            need_replan = True

        if need_replan:
            print("Replanning...")
            X_ref, U_ref = ilqr_plan(current_state, goal, obstacles)
            step_idx = 0
            ilqr_step_counter = 0

        # --- apply next iLQR control ---
        if ilqr_step_counter == 0:
            v, w = U_ref[step_idx]
            step_idx += 1

            # Convert (v, w) → wheel speeds
            # Sample uncertain parameters once per simulation
            r_mean = 0.05 #mean of ideal parameter
            L_mean = 0.20

            r_std = 0.002      # 2 mm std dev
            L_std = 0.005      # 5 mm std dev

            r = np.random.normal(r_mean, r_std)
            L = np.random.normal(L_mean, L_std)
            omega_r = (v + (L/2)*w) / r
            omega_l = (v - (L/2)*w) / r

        ilqr_step_counter = (ilqr_step_counter + 1) % ilqr_dt_steps

        # --- apply wheel speeds ---
        apply_inputs_Brownian(robotId, omega_l, omega_r,ilqr_dt)

        # --- get actual state after applying inputs ---
        pos, orn = p.getBasePositionAndOrientation(robotId)
        x_bot, y_bot, _ = pos
        yaw_bot = p.getEulerFromQuaternion(orn)[2]
        x = np.array([x_bot, y_bot, yaw_bot])

        # --- actual control used by planner ---
        u = np.array([v, w])

        # --- log cost (normal step, no terminal cost) ---
        t = i * sim_dt
        log_states_and_time(log, x, t,)
        p.stepSimulation()
        time.sleep(sim_dt)

        # --- camera ---
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=[x_bot, y_bot, 0.1]
        )


    p.disconnect()

    # show_plots=True to show, False to suppress
    plot_logs(log, show_plots=False)
    return log

if __name__ == "__main__":
    startpos=[-1,0,0]
    t_end = main(startpos)
