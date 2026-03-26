import numpy as np
import math

def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

def diffdrive_dynamics(x, u, dt):
    px, py, theta = x
    v, omega = u

    px_next = px + dt * v * math.cos(theta)
    py_next = py + dt * v * math.sin(theta)
    theta_next = wrap_angle(theta + dt * omega)

    return np.array([px_next, py_next, theta_next])

def cost_function(x, u, goal, obstacles, Q, R, obs_weight):
    dx = x[0] - goal[0]
    dy = x[1] - goal[1]

    state_cost = Q[0]*dx*dx + Q[1]*dy*dy
    control_cost = R[0]*u[0]*u[0] + R[1]*u[1]*u[1]

    obs_cost = 0.0
    for (ox, oy, orad) in obstacles:
        dist = math.hypot(x[0] - ox, x[1] - oy)
        safe_dist = orad + 0.5
        if dist < safe_dist:
            obs_cost += obs_weight * (safe_dist - dist)**2

    return state_cost + control_cost + obs_cost

def ilqr_plan(start_state, goal, obstacles,
              N=200, dt=0.05,
              Q=np.array([70.0, 70.0]),
              R=np.array([0.2, 0.2]),
              Qf=np.array([2000.0, 2000.0]),
              obs_weight=1000.0,
              max_iters=50):

    n_x = 3
    n_u = 2

    # INITIAL GUESS: straight line toward goal
    dx = goal[0] - start_state[0]
    dy = goal[1] - start_state[1]
    desired_theta = math.atan2(dy, dx)

    dtheta = wrap_angle(desired_theta - start_state[2])

    # gentle turn over the whole horizon
    omega_guess = dtheta / (N * dt)

    # modest forward velocity, avoids singularity points
    v_guess = 0.3
    U = np.zeros((N, n_u))
    U[:, 0] = v_guess
    U[:, 1] = omega_guess

    # ---------------------------------------------------------
    # rollout function
    # ---------------------------------------------------------
    def rollout(x0, U):
        X = np.zeros((N+1, n_x))
        X[0] = x0
        for k in range(N):
            X[k+1] = diffdrive_dynamics(X[k], U[k], dt)
        return X

    X = rollout(start_state, U)

    # ---------------------------------------------------------
    # iLQR iterations
    # ---------------------------------------------------------
    for it in range(max_iters):

        Vx = np.zeros(n_x)
        Vxx = np.zeros((n_x, n_x))

        # terminal cost (position only)
        error_x = X[-1,0] - goal[0]
        error_y = X[-1,1] - goal[1]

        Vx[0] = 2*Qf[0]*error_x
        Vx[1] = 2*Qf[1]*error_y
        Vxx[0,0] = 2*Qf[0]
        Vxx[1,1] = 2*Qf[1]

        K = np.zeros((N, n_u, n_x))
        k_ff = np.zeros((N, n_u))

        # -----------------------------------------------------
        # backward pass
        # -----------------------------------------------------
        for k in reversed(range(N)):
            xk = X[k]
            uk = U[k]

            theta = xk[2]
            v = uk[0]

            A = np.array([
                [1.0, 0.0, -dt * v * math.sin(theta)],
                [0.0, 1.0,  dt * v * math.cos(theta)],
                [0.0, 0.0,  1.0]
            ])
            B = np.array([
                [dt * math.cos(theta), 0.0],
                [dt * math.sin(theta), 0.0],
                [0.0,               dt]
            ])

            lx = np.zeros(n_x)
            lu = np.zeros(n_u)
            lxx = np.zeros((n_x, n_x))
            luu = np.zeros((n_u, n_u))
            lux = np.zeros((n_u, n_x))

            dxg = xk[0] - goal[0]
            dyg = xk[1] - goal[1]

            lx[0] += 2*Q[0]*dxg
            lx[1] += 2*Q[1]*dyg
            lxx[0,0] += 2*Q[0]
            lxx[1,1] += 2*Q[1]

            lu[0] += 2*R[0]*uk[0]
            lu[1] += 2*R[1]*uk[1]
            luu[0,0] += 2*R[0]
            luu[1,1] += 2*R[1]

            # obstacle gradient
            for (ox, oy, orad) in obstacles:
                dist = math.hypot(xk[0] - ox, xk[1] - oy)
                safe_dist = orad + 0.5
                if dist < safe_dist and dist > 1e-4:
                    dcost = 2*obs_weight*(safe_dist - dist)*(-1.0/dist)
                    lx[0] += dcost * (xk[0] - ox)
                    lx[1] += dcost * (xk[1] - oy)

            Qx  = lx  + A.T @ Vx
            Qu  = lu  + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu_reg = Quu + 1e-6 * np.eye(n_u)
            Quu_inv = np.linalg.inv(Quu_reg)

            K[k] = -Quu_inv @ Qux
            k_ff[k] = -Quu_inv @ Qu

            Vx = Qx + K[k].T @ Quu @ k_ff[k] + K[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            Vxx = 0.5 * (Vxx + Vxx.T)

        # -----------------------------------------------------
        # forward pass with line search
        # -----------------------------------------------------
        alpha_list = [1.0, 0.5, 0.25, 0.1]
        best_cost = float('inf')
        best_X = None
        best_U = None

        for alpha in alpha_list:
            X_new = np.zeros_like(X)
            U_new = np.zeros_like(U)
            X_new[0] = start_state
            cost_total = 0.0

            for k in range(N):
                du = alpha * k_ff[k] + K[k] @ (X_new[k] - X[k])
                U_new[k] = U[k] + du

                U_new[k,0] = np.clip(U_new[k,0], 0.0, 1.0)
                U_new[k,1] = np.clip(U_new[k,1], -2.0, 2.0)

                X_new[k+1] = diffdrive_dynamics(X_new[k], U_new[k], dt)
                cost_total += cost_function(X_new[k], U_new[k], goal, obstacles, Q, R, obs_weight)

            dx = X_new[-1,0] - goal[0]
            dy = X_new[-1,1] - goal[1]
            cost_total += Qf[0]*dx*dx + Qf[1]*dy*dy

            if cost_total < best_cost:
                best_cost = cost_total
                best_X = X_new
                best_U = U_new

        # fallback if line search fails
        if best_X is None:
            best_X = X
            best_U = U

        X = best_X
        U = best_U

    return X, U
