import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import matplotlib.pyplot as plt

# ============================================================
# 1. URDF: simple 2D drone (no baked-in rotation)
# ============================================================

DRONE_URDF = r"""<?xml version="1.0" ?>
<robot name="drone2d">
  <link name="base_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.02" iyy="0.02" izz="0.02" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.05 0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.1 0.1 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <box size="0.3 0.05 0.05"/>
      </geometry>
    </collision>
  </link>

  <link name="rotor_left">
    <visual>
      <origin xyz="-0.15 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.02" radius="0.03"/>
      </geometry>
      <material name="red">
        <color rgba="0.8 0.1 0.1 1"/>
      </material>
    </visual>
  </link>

  <link name="rotor_right">
    <visual>
      <origin xyz="0.15 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.02" radius="0.03"/>
      </geometry>
      <material name="green">
        <color rgba="0.1 0.8 0.1 1"/>
      </material>
    </visual>
  </link>

  <joint name="joint_left" type="fixed">
    <parent link="base_link"/>
    <child link="rotor_left"/>
    <origin xyz="-0.15 0 0" rpy="0 0 0"/>
  </joint>

  <joint name="joint_right" type="fixed">
    <parent link="base_link"/>
    <child link="rotor_right"/>
    <origin xyz="0.15 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""

URDF_PATH = "drone2d.urdf"
with open(URDF_PATH, "w") as f:
    f.write(DRONE_URDF)

# ============================================================
# 2. Physical parameters
# ============================================================

m = 1.0
I = 0.02
l = 0.15
g = 9.81

# ============================================================
# 3. Nonlinear dynamics (x–z plane, z up, yaw = theta)
# ============================================================

# State: [x, z, theta, xdot, zdot, w]
# Controls: [u1, u2] (left/right rotor thrusts)
def drone_dynamics(x, u):
    x_pos, z_pos, th, xdot, zdot, w = x
    u1, u2 = u

    T = u1 + u2
    tau = l * (u2 - u1)

    # Thrust along body +Z, rotated into world x–z
    xddot = -(T / m) * np.sin(th)
    zddot =  (T / m) * np.cos(th) - g
    wdot  = tau / I

    return np.array([xdot, zdot, w, xddot, zddot, wdot])

def linearize_dynamics(x, u, eps=1e-5):
    n = len(x)
    m_in = len(u)
    A = np.zeros((n, n))
    B = np.zeros((n, m_in))

    for i in range(n):
        dx = np.zeros(n); dx[i] = eps
        A[:, i] = (drone_dynamics(x + dx, u) - drone_dynamics(x - dx, u)) / (2 * eps)

    for j in range(m_in):
        du = np.zeros(m_in); du[j] = eps
        B[:, j] = (drone_dynamics(x, u + du) - drone_dynamics(x, u - du)) / (2 * eps)

    return A, B

# ============================================================
# 4. Discrete dynamics
# ============================================================

def f_discrete(x, u, dt):
    return x + dt * drone_dynamics(x, u)

def linearize_discrete(x, u, dt):
    A, B = linearize_dynamics(x, u)
    return np.eye(len(x)) + dt * A, dt * B

# ============================================================
# 5. Cost
# ============================================================

def running_cost(x, u, x_goal, Q, R):
    dx = x - x_goal
    return dx.T @ Q @ dx + u.T @ R @ u

def terminal_cost(x, x_goal, Qf):
    dx = x - x_goal
    return dx.T @ Qf @ dx

# ============================================================
# 6. Rollout
# ============================================================

def rollout(x0, U, dt, x_goal, Q, R, Qf):
    N = len(U)
    X = np.zeros((N + 1, len(x0)))
    X[0] = x0
    cost = 0.0

    for k in range(N):
        cost += running_cost(X[k], U[k], x_goal, Q, R)
        X[k+1] = f_discrete(X[k], U[k], dt)

    cost += terminal_cost(X[-1], x_goal, Qf)
    return X, cost

# ============================================================
# 7. iLQR
# ============================================================

def ilqr(x0, U_init, dt, x_goal, Q, R, Qf, max_iters=50):
    U = U_init.copy()
    X, J = rollout(x0, U, dt, x_goal, Q, R, Qf)
    N = len(U)
    n = len(x0)
    m_in = U.shape[1]

    for it in range(max_iters):
        Vx  = 2 * Qf @ (X[-1] - x_goal)
        Vxx = 2 * Qf.copy()

        K = np.zeros((N, m_in, n))
        kff = np.zeros((N, m_in))

        # Backward pass
        for t in reversed(range(N)):
            Ad, Bd = linearize_discrete(X[t], U[t], dt)

            lx  = 2 * Q @ (X[t] - x_goal)
            lu  = 2 * R @ U[t]
            lxx = 2 * Q
            luu = 2 * R
            lux = np.zeros((m_in, n))

            Qx  = lx + Ad.T @ Vx
            Qu  = lu + Bd.T @ Vx
            Qxx = lxx + Ad.T @ Vxx @ Ad
            Quu = luu + Bd.T @ Vxx @ Bd
            Qux = lux + Bd.T @ Vxx @ Ad

            Quu_inv = np.linalg.inv(Quu + 1e-6 * np.eye(m_in))

            K[t]   = -Quu_inv @ Qux
            kff[t] = -Quu_inv @ Qu

            Vx  = Qx + K[t].T @ Quu @ kff[t] + K[t].T @ Qu + Qux.T @ kff[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]
            Vxx = 0.5 * (Vxx + Vxx.T)

        # Forward line search
        improved = False
        for alpha in [1.0, 0.5, 0.25, 0.1]:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = x0

            for t in range(N):
                dx = X_new[t] - X[t]
                du = alpha * kff[t] + K[t] @ dx
                U_new[t] = U[t] + du
                X_new[t+1] = f_discrete(X_new[t], U_new[t], dt)

            J_new = sum(running_cost(X_new[t], U_new[t], x_goal, Q, R) for t in range(N))
            J_new += terminal_cost(X_new[-1], x_goal, Qf)

            if J_new < J:
                U, X, J = U_new, X_new, J_new
                improved = True
                break

        if not improved:
            break

    return X, U, J

# ============================================================
# 8. PyBullet 2D environment (x–z plane, z up)
# ============================================================

class Drone2DEnv:
    def __init__(self, gui=True, dt=1/240):
        self.dt = dt

        if gui:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # z is up in PyBullet
        p.setGravity(0, 0, -g)

        # Side view: x right, z up, y into screen
        p.resetDebugVisualizerCamera(
            cameraDistance=3.0,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 1.0, 0]
        )

        self.plane = p.loadURDF("plane.urdf")
        self.drone = p.loadURDF(URDF_PATH, basePosition=[0,1,0])

        p.changeDynamics(self.drone, -1, linearDamping=0.1, angularDamping=0.1)

    def reset(self, pos=[0,1], angle=0):
        p.resetBasePositionAndOrientation(
            self.drone,
            [pos[0], 0, pos[1]],  # x, y, z  (y into screen)
            p.getQuaternionFromEuler([0, 0, angle])
        )
        p.resetBaseVelocity(self.drone, [0,0,0], [0,0,0])
        return self.get_state()

    def get_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        vel, ang = p.getBaseVelocity(self.drone)

        x, y, z = pos
        xdot, ydot, zdot = vel
        _, _, th = p.getEulerFromQuaternion(orn)
        _, _, w = ang

        return np.array([x, z, th, xdot, zdot, w])

    def step(self, u):
        u1, u2 = u
        T = u1 + u2
        tau = l * (u2 - u1)

        # Thrust along body +Z, rotated into world
        p.applyExternalForce(self.drone, -1, [0, 0, T], [0,0,0], p.LINK_FRAME)
        # Torque around body Z
        p.applyExternalTorque(self.drone, -1, [0,0,tau], p.LINK_FRAME)

        p.stepSimulation()
        time.sleep(self.dt)
        return self.get_state()

# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    dt = 0.02
    N = 200

    x0 = np.array([0, 0, 0.1, 0, 0, 0])      # [x, z, theta, xdot, zdot, w]
    x_goal = np.array([1, 1, 0, 0, 0, 0])

    u_hover = np.array([0.5*m*g, 0.5*m*g])
    U_init = np.tile(u_hover, (N, 1))

    Q  = np.diag([50,50,5,10,10,1])
    Qf = np.diag([2000,2000,20,10,10,10])
    R  = 0.5 * np.eye(2)

    print("Running iLQR...")
    X_opt, U_opt, J_opt = ilqr(x0, U_init, dt, x_goal, Q, R, Qf)
    print("Final cost:", J_opt)
    print("Final state:", X_opt[-1])

    # Plots (x–z plane)
    t = np.arange(X_opt.shape[0]) * dt

    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    fig.suptitle("iLQR 2D Drone Results (x–z plane)")

    axs[0].plot(X_opt[:, 0], X_opt[:, 1], label="trajectory")
    axs[0].scatter([x0[0]], [x0[1]], color="green", label="start")
    axs[0].scatter([x_goal[0]], [x_goal[1]], color="red", label="goal")
    axs[0].set_xlabel("x (m)")
    axs[0].set_ylabel("z (m)")
    axs[0].set_title("Position in 2D (x–z)")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, X_opt[:, 2])
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("theta (rad)")
    axs[1].set_title("Angle")
    axs[1].grid(True)

    axs[2].plot(t[:-1], U_opt[:, 0], label="u1 (left thrust)")
    axs[2].plot(t[:-1], U_opt[:, 1], label="u2 (right thrust)")
    axs[2].set_xlabel("time (s)")
    axs[2].set_ylabel("thrust (N)")
    axs[2].set_title("Control Inputs")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

    # PyBullet playback
    env = Drone2DEnv(gui=True, dt=1/240)
    env.reset([x0[0], x0[1]], x0[2])

    steps = max(1, int(dt / env.dt))
    print("Playing trajectory in PyBullet...")
    for u in U_opt:
        for _ in range(steps):
            env.step(u)

    input("Press Enter to exit...")
    p.disconnect()