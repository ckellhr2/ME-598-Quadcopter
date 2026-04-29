import matplotlib.pyplot as plt

def plot_logs(log, show_plots=True):
    if not show_plots:
        return

    # Extract arrays
    xs = [e["x"] for e in log]
    ys = [e["y"] for e in log]
    zs = [e["z"] for e in log]

    rolls  = [e["roll"] for e in log]
    pitches = [e["pitch"] for e in log]
    yaws   = [e["yaw"] for e in log]

    vxs = [e["vx"] for e in log]
    vys = [e["vy"] for e in log]
    vzs = [e["vz"] for e in log]

    wxs = [e["wx"] for e in log]
    wys = [e["wy"] for e in log]
    wzs = [e["wz"] for e in log]

    forces  = [e["force"] for e in log]
    torques = [e["torque"] for e in log]

    t = range(len(log))

    # --- Trajectory ---
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("XY Trajectory")
    plt.axis("equal")

    # --- Yaw ---
    plt.figure()
    plt.plot(t, yaws)
    plt.xlabel("timestep")
    plt.ylabel("yaw (rad)")
    plt.title("Yaw over time")

    # --- Linear velocities ---
    plt.figure()
    plt.plot(t, vxs, label="vx")
    plt.plot(t, vys, label="vy")
    plt.plot(t, vzs, label="vz")
    plt.xlabel("timestep")
    plt.ylabel("linear velocity")
    plt.legend()
    plt.title("Linear Velocities")

    # --- Angular velocities ---
    plt.figure()
    plt.plot(t, wxs, label="wx")
    plt.plot(t, wys, label="wy")
    plt.plot(t, wzs, label="wz")
    plt.xlabel("timestep")
    plt.ylabel("angular velocity")
    plt.legend()
    plt.title("Angular Velocities")

    # --- Inputs ---
    plt.figure()
    plt.plot(t, forces, label="force")
    plt.plot(t, torques, label="torque")
    plt.xlabel("timestep")
    plt.ylabel("input")
    plt.legend()
    plt.title("Control Inputs")

    plt.show()