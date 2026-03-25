import pybullet as p
import os
import time
import pybullet_data
import math
import numpy as np
import matplotlib.pyplot as plt
from DronePlots import plot_logs
from ApplyInputsNoisy import apply_inputs_noisy
from ideal_iLQRController import ilqr_plan


log = {
    "t": [],
    "x": [],
    "u": [],
    "cost_per_step": [],
    "cost_cumulative": 0.0
}
tolerance=0.2 #tolerance for goal checking to see if position has been reached

def log_actual_cost(
    log, x, u, t,
    goal, obstacles,terminal=False,
    Q=np.array([70.0, 70.0]), R=np.array([1, 1]), Qf=np.array([2000.0, 2000.0]),
    obs_weight=1000.0,
    dt=0.05,
):
    #Logs actual state, control, and computes the SAME cost used by the planner.

    # --- Log trajectory ---
    log["t"].append(t)
    log["x"].append(x.copy())
    log["u"].append(u.copy())

    # --- Stage cost (same as cost_function) ---
    dx = x[0] - goal[0]
    dy = x[1] - goal[1]

    state_cost = Q[0] * dx * dx + Q[1] * dy * dy
    control_cost = R[0] * u[0] * u[0] + R[1] * u[1] * u[1]

    # --- Obstacle cost (identical to planner) ---
    obs_cost = 0.0
    for (ox, oy, orad) in obstacles:
        dist = math.hypot(x[0] - ox, x[1] - oy)
        safe_dist = orad + 0.75
        if dist < safe_dist:
            obs_cost += obs_weight * (safe_dist - dist)**2

    # --- Terminal cost (only at final step) ---
    if terminal:
        term_cost = Qf[0] * dx * dx + Qf[1] * dy * dy
    else:
        term_cost = 0.0

    # --- Total instantaneous cost ---
    step_cost = state_cost + control_cost + obs_cost + term_cost
    log["cost_per_step"].append(step_cost)

    # --- Accumulate cost over time ---
    log["cost_cumulative"] += step_cost * dt

def main(goal = [10.0, 15.0],obstacle1_position = [2,2],
         obstacle2_position = [3,1],obstacle3_position = [8,1]):
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    planeId = p.loadURDF("plane.urdf")
    startPos = [0, 0, 0.1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])

    urdf_path = os.path.join(os.path.dirname(__file__), "diff_drive.urdf")
    robotId = p.loadURDF(urdf_path, startPos, startOrientation)

    

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
    obstacle2_position = [3,1]
    obstacle3_position = [8,1]
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

    #Q matrix values: Real Q lives in controller but values needed for cost logging

    # PyBullet runs at 240 Hz, iLQR at dt = 0.05 → 20 Hz
    sim_dt = 1/240
    ilqr_dt = 0.05
    ilqr_dt_steps = int(ilqr_dt / sim_dt)   # = 12

    ilqr_step_counter = 0
    step_idx = 0

    # --- initial plan ---
    X_ref, U_ref = ilqr_plan(start_state, goal, obstacles)
    terminal = False #used to tell logger that this step is the last if true for terminal cost
    
    for i in range(10000):

        # --- get current robot state ---
        pos, orn = p.getBasePositionAndOrientation(robotId)
        x_bot, y_bot, _ = pos
        yaw_bot = p.getEulerFromQuaternion(orn)[2]
        current_state = np.array([x_bot, y_bot, yaw_bot])

        # --- stopping condition ---
        dist_to_goal = math.hypot(x_bot - goal[0], y_bot - goal[1])
        if dist_to_goal < tolerance:
            terminal = True
            omega_l = 0.0
            omega_r = 0.0
            apply_inputs_noisy(robotId, omega_l, omega_r)
            #get final state for cost logging
            pos, orn = p.getBasePositionAndOrientation(robotId)
            x_bot, y_bot, _ = pos
            yaw_bot = p.getEulerFromQuaternion(orn)[2]
            x = np.array([x_bot, y_bot, yaw_bot])
            #know it is final step so inputs forced to 0
            u=[0,0]
            t=i*ilqr_dt
            log_actual_cost(log, x, u, t,goal, obstacles, terminal,
                            Q=np.array([70.0, 70.0]), R=np.array([1, 1]), Qf=np.array([2000.0, 2000.0]),
                            obs_weight=1000.0,dt=0.05,)
            print("Goal reached — stopping.")
            break

        # setting up conditions
        need_replan = False #used to call planner again if true
       
        
        # 1. End of horizon
        if step_idx >= len(U_ref):
            need_replan = True

        # 2. Robot drifted too far from reference
        ref_x, ref_y = X_ref[min(step_idx, len(X_ref)-1), :2]
        drift = math.hypot(x_bot - ref_x, y_bot - ref_y)
        if drift > 0.25:   # you can tune this
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
            r = 0.05
            L = 0.20
            omega_r = (v + (L/2)*w) / r
            omega_l = (v - (L/2)*w) / r

        ilqr_step_counter = (ilqr_step_counter + 1) % ilqr_dt_steps

        # --- apply wheel speeds ---
        apply_inputs_noisy(robotId, omega_l, omega_r)

        # --- get actual state after applying inputs ---
        pos, orn = p.getBasePositionAndOrientation(robotId)
        x_bot, y_bot, _ = pos
        yaw_bot = p.getEulerFromQuaternion(orn)[2]
        x = np.array([x_bot, y_bot, yaw_bot])

        # --- actual control used by planner ---
        u = np.array([v, w])

        # --- log cost (normal step, no terminal cost) ---
        t = i * ilqr_dt
        log_actual_cost(log, x, u, t,
            goal, obstacles, terminal,
            Q=np.array([70.0, 70.0]),
            R=np.array([1, 1]),
            Qf=np.array([2000.0, 2000.0]),
            obs_weight=1000.0,
            dt=0.05)

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
    return log["cost_cumulative"]

if __name__ == "__main__":
    goal = [1,1]
    obstacle1_position = [2,2]
    obstacle2_position = [3,1]
    obstacle3_position = [8,1]
    total_cost = main(goal,obstacle1_position,obstacle2_position,obstacle3_position)
    print("Total cost:", total_cost)