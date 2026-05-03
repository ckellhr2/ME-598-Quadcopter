import numpy as np
import matplotlib

# ============================
# Physical parameters
# ============================
m = 1.0      # mass (kg)
I = 0.02     # moment of inertia (kg*m^2)
l = 0.2      # half-arm length (m)
g = 9.81     # gravity (m/s^2)

# ============================
# Dynamics
# State: [x, y, theta, xdot, ydot, omega]
# Input: [u1, u2]
# ============================
def drone_dynamics(x, u):
    x_pos, y_pos, th, xdot, ydot, w = x
    u1, u2 = u

    T = u1 + u2
    tau = l * (u2 - u1)

    xddot = -(T / m) * np.sin(th)
    yddot =  (T / m) * np.cos(th) - g
    wdot  = tau / I

    return np.array([
        xdot,
        ydot,
        w,
        xddot,
        yddot,
        wdot
    ])

def linearize_dynamics(x, u, eps=1e-5):
    n = len(x)
    m_in = len(u)

    A = np.zeros((n, n))
    B = np.zeros((n, m_in))

    fx = drone_dynamics(x, u)

    # A = df/dx
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        f_plus  = drone_dynamics(x + dx, u)
        f_minus = drone_dynamics(x - dx, u)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    # B = df/du
    for j in range(m_in):
        du = np.zeros(m_in)
        du[j] = eps
        f_plus  = drone_dynamics(x, u + du)
        f_minus = drone_dynamics(x, u - du)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B

# ============================
# Discrete-time dynamics
# x_{k+1} = x_k + dt * f(x_k, u_k)
# ============================
def f_discrete(x, u, dt):
    return x + dt * drone_dynamics(x, u)

def linearize_discrete(x, u, dt):
    A, B = linearize_dynamics(x, u)
    n = A.shape[0]
    Ad = np.eye(n) + dt * A
    Bd = dt * B
    return Ad, Bd

# ============================
# Cost function
# ============================
def running_cost(x, u, x_goal, Q, R):
    dx = x - x_goal
    return dx.T @ Q @ dx + u.T @ R @ u

def terminal_cost(x, x_goal, Qf):
    dx = x - x_goal
    return dx.T @ Qf @ dx

# ============================
# Rollout
# ============================
def rollout(x0, U, dt, x_goal, Q, R, Qf):
    N = U.shape[0]
    n = x0.shape[0]

    X = np.zeros((N + 1, n))
    X[0] = x0
    cost = 0.0

    for k in range(N):
        xk = X[k]
        uk = U[k]
        cost += running_cost(xk, uk, x_goal, Q, R)
        X[k + 1] = f_discrete(xk, uk, dt)

    cost += terminal_cost(X[-1], x_goal, Qf)
    return X, cost

# ============================
# iLQR
# ============================
def ilqr(
    x0,
    U_init,
    dt,
    x_goal,
    Q,
    R,
    Qf,
    max_iters=50,
    alpha_list=None
):
    if alpha_list is None:
        alpha_list = [1.0, 0.5, 0.25, 0.1, 0.05]

    N, m_in = U_init.shape
    n = x0.shape[0]

    U = U_init.copy()
    X, J = rollout(x0, U, dt, x_goal, Q, R, Qf)

    for it in range(max_iters):
        # Backward pass
        Vx  = 2 * Qf @ (X[-1] - x_goal)
        Vxx = 2 * Qf.copy()

        K = np.zeros((N, m_in, n))
        k = np.zeros((N, m_in))

        for k_idx in reversed(range(N)):
            xk = X[k_idx]
            uk = U[k_idx]

            Ad, Bd = linearize_discrete(xk, uk, dt)

            # Quadratic expansion of cost
            lx  = 2 * Q @ (xk - x_goal)
            lu  = 2 * R @ uk
            lxx = 2 * Q
            luu = 2 * R
            lux = np.zeros((m_in, n))

            Qx  = lx + Ad.T @ Vx
            Qu  = lu + Bd.T @ Vx
            Qxx = lxx + Ad.T @ Vxx @ Ad
            Quu = luu + Bd.T @ Vxx @ Bd
            Qux = lux + Bd.T @ Vxx @ Ad

            # Regularization (simple)
            reg = 1e-6 * np.eye(m_in)
            Quu_reg = Quu + reg

            # Gains
            Quu_inv = np.linalg.inv(Quu_reg)
            K[k_idx] = -Quu_inv @ Qux
            k[k_idx] = -Quu_inv @ Qu

            # Value function update
            Vx  = Qx + K[k_idx].T @ Quu @ k[k_idx] + K[k_idx].T @ Qu + Qux.T @ k[k_idx]
            Vxx = Qxx + K[k_idx].T @ Quu @ K[k_idx] + K[k_idx].T @ Qux + Qux.T @ K[k_idx]
            # Symmetrize Vxx
            Vxx = 0.5 * (Vxx + Vxx.T)

        # Forward line search
        improved = False
        for alpha in alpha_list:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = x0

            for k_idx in range(N):
                dx = X_new[k_idx] - X[k_idx]
                du = k[k_idx] * alpha + K[k_idx] @ dx
                U_new[k_idx] = U[k_idx] + du
                X_new[k_idx + 1] = f_discrete(X_new[k_idx], U_new[k_idx], dt)

            J_new = 0.0
            for k_idx in range(N):
                J_new += running_cost(X_new[k_idx], U_new[k_idx], x_goal, Q, R)
            J_new += terminal_cost(X_new[-1], x_goal, Qf)

            if J_new < J:
                U = U_new
                X = X_new
                J = J_new
                improved = True
                break

        if not improved:
            # No improvement for any alpha
            break

    return X, U, J

# ============================
# Example usage
# ============================
if __name__ == "__main__":
    dt = 0.02
    N = 200

    # Initial state: at origin, small angle
    x0 = np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0])

    # Goal state
    x_goal = np.array([2.0, 7.0, 0.0, 0.0, 0.0, 0.0])

    # Hover thrust (split between two rotors)
    u_hover = np.array([0.5 * m * g, 0.5 * m * g])

    # Initial control sequence: hover
    U_init = np.tile(u_hover, (N, 1))

    # Cost weights
    Q  = np.diag([10.0, 10.0, 5.0, 1.0, 1.0, 1.0])
    Qf = np.diag([50.0, 50.0, 10.0, 5.0, 5.0, 5.0])
    R  = 0.1 * np.eye(2)

    X_opt, U_opt, J_opt = ilqr(
        x0=x0,
        U_init=U_init,
        dt=dt,
        x_goal=x_goal,
        Q=Q,
        R=R,
        Qf=Qf,
        max_iters=50
    )

    print("Final cost:", J_opt)
    print("Final state:", X_opt[-1])

    import matplotlib.pyplot as plt

# ============================
# Plotting
# ============================

t = np.arange(X_opt.shape[0]) * dt

fig, axs = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle("iLQR Drone Trajectory")

# --- Position plot ---
axs[0].plot(X_opt[:, 0], X_opt[:, 1], label="trajectory")
axs[0].scatter([x0[0]], [x0[1]], color="green", label="start")
axs[0].scatter([x_goal[0]], [x_goal[1]], color="red", label="goal")
axs[0].set_xlabel("x position (m)")
axs[0].set_ylabel("y position (m)")
axs[0].set_title("Position in 2D")
axs[0].legend()
axs[0].grid(True)

# --- Angle plot ---
axs[1].plot(t, X_opt[:, 2])
axs[1].set_xlabel("time (s)")
axs[1].set_ylabel("theta (rad)")
axs[1].set_title("Angle over time")
axs[1].grid(True)

# --- Control inputs ---
axs[2].plot(t[:-1], U_opt[:, 0], label="u1 (left thrust)")
axs[2].plot(t[:-1], U_opt[:, 1], label="u2 (right thrust)")
axs[2].set_xlabel("time (s)")
axs[2].set_ylabel("thrust (N)")
axs[2].set_title("Control inputs")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()