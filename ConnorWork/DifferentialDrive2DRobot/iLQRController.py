import numpy as np
import math

def diffdrive_dynamics(x, u, dt):
    # x = [x, y, theta], u = [v, w]
    px, py, th = x
    v, w = u

    px_next = px + dt * v * math.cos(th)
    py_next = py + dt * v * math.sin(th)
    th_next = th + dt * w
    return np.array([px_next, py_next, th_next])

def cost_function(x, u, goal, obstacles, Q, R, obs_weight):
    # state cost
    dx = x[0] - goal[0]
    dy = x[1] - goal[1]
    state_cost = Q[0]*dx*dx + Q[1]*dy*dy

    # control cost
    control_cost = R[0]*u[0]*u[0] + R[1]*u[1]*u[1]

    # obstacle cost
    obs_cost = 0.0
    for (ox, oy, orad) in obstacles:
        dist = math.hypot(x[0] - ox, x[1] - oy)
        # soft barrier around obstacle
        safe_dist = orad + 0.2
        if dist < safe_dist:
            obs_cost += obs_weight * (safe_dist - dist)**2

    return state_cost + control_cost + obs_cost

def ilqr_plan(start_state, goal, obstacles,
              N=400, dt=0.05,
              Q=np.array([10.0, 10.0]),
              R=np.array([0.1, 0.1]),
              Qf=np.array([200.0, 200.0]),
              obs_weight=200.0,
              max_iters=50):
    """
    iLQR for diff-drive with state x=[x,y,theta], control u=[v,w].
    Returns arrays X (N+1,3), U (N,2).
    """
    n_x = 3
    n_u = 2

    # initial guess: straight-line, small forward velocity
    U = np.zeros((N, n_u))
    U[:, 0] = 0.5  # v
    U[:, 1] = 0.0  # w

    def rollout(x0, U):
        X = np.zeros((N+1, n_x))
        X[0] = x0
        for k in range(N):
            X[k+1] = diffdrive_dynamics(X[k], U[k], dt)
        return X

    X = rollout(start_state, U)

    for it in range(max_iters):
        # backward pass
        Vx = np.zeros(n_x)
        Vxx = np.zeros((n_x, n_x))

        # terminal cost (only position)
        dx = X[-1,0] - goal[0]
        dy = X[-1,1] - goal[1]
        Vx[0] = 2*Qf[0]*dx
        Vx[1] = 2*Qf[1]*dy
        Vxx[0,0] = 2*Qf[0]
        Vxx[1,1] = 2*Qf[1]

        K = np.zeros((N, n_u, n_x))
        k_ff = np.zeros((N, n_u))

        for k in reversed(range(N)):
            xk = X[k]
            uk = U[k]

            th = xk[2]
            v = uk[0]
            w = uk[1]

            # linearized dynamics: x_{k+1} = f + A dx + B du
            A = np.array([
                [1.0, 0.0, -dt * v * math.sin(th)],
                [0.0, 1.0,  dt * v * math.cos(th)],
                [0.0, 0.0,  1.0]
            ])
            B = np.array([
                [dt * math.cos(th), 0.0],
                [dt * math.sin(th), 0.0],
                [0.0,               dt]
            ])

            # cost derivatives
            lx = np.zeros(n_x)
            lu = np.zeros(n_u)
            lxx = np.zeros((n_x, n_x))
            luu = np.zeros((n_u, n_u))
            lux = np.zeros((n_u, n_x))

            # state cost (running)
            dxg = xk[0] - goal[0]
            dyg = xk[1] - goal[1]
            lx[0] += 2*Q[0]*dxg
            lx[1] += 2*Q[1]*dyg
            lxx[0,0] += 2*Q[0]
            lxx[1,1] += 2*Q[1]

            # control cost
            lu[0] += 2*R[0]*uk[0]
            lu[1] += 2*R[1]*uk[1]
            luu[0,0] += 2*R[0]
            luu[1,1] += 2*R[1]

            # obstacle cost (approximate gradient)
            for (ox, oy, orad) in obstacles:
                dist = math.hypot(xk[0] - ox, xk[1] - oy)
                safe_dist = orad + 0.2
                if dist < safe_dist and dist > 1e-4:
                    dcost = 2*obs_weight*(safe_dist - dist)*(-1.0/dist)
                    lx[0] += dcost * (xk[0] - ox)
                    lx[1] += dcost * (xk[1] - oy)

            # Q-function expansion
            Qx  = lx  + A.T @ Vx
            Qu  = lu  + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            # regularization for numerical stability
            Quu_reg = Quu + 1e-6 * np.eye(n_u)

            # feedback + feedforward gains
            Quu_inv = np.linalg.inv(Quu_reg)
            K[k] = -Quu_inv @ Qux
            k_ff[k] = -Quu_inv @ Qu

            # value function update
            Vx = Qx + K[k].T @ Quu @ k_ff[k] + K[k].T @ Qu + Qux.T @ k_ff[k]
            Vxx = Qxx + K[k].T @ Quu @ K[k] + K[k].T @ Qux + Qux.T @ K[k]
            Vxx = 0.5 * (Vxx + Vxx.T)  # symmetrize

        # forward pass with line search
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
                # simple bounds on v,w
                U_new[k,0] = np.clip(U_new[k,0], 0.0, 1.0)
                U_new[k,1] = np.clip(U_new[k,1], -2.0, 2.0)
                X_new[k+1] = diffdrive_dynamics(X_new[k], U_new[k], dt)
                cost_total += cost_function(X_new[k], U_new[k], goal, obstacles, Q, R, obs_weight)

            # terminal cost
            dx = X_new[-1,0] - goal[0]
            dy = X_new[-1,1] - goal[1]
            cost_total += Qf[0]*dx*dx + Qf[1]*dy*dy

            if cost_total < best_cost:
                best_cost = cost_total
                best_X = X_new
                best_U = U_new

        X = best_X
        U = best_U

    return X, U