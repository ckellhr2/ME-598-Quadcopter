def goal_controller(body_id, goal):
    # Robot pose
    pos, orn = p.getBasePositionAndOrientation(body_id)
    x, y, _ = pos
    yaw = p.getEulerFromQuaternion(orn)[2]

    gx, gy = goal
    dx = gx - x
    dy = gy - y

    desired_yaw = math.atan2(dy, dx)

    # heading error wrapped to [-pi, pi]
    yaw_error = desired_yaw - yaw
    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi

    # Gains
    k_turn = 5
    k_drive = 2

    # Differential drive mapping
    # forward speed proportional to cos(error)
    v = k_drive * max(0.0, math.cos(yaw_error))
    w = k_turn * yaw_error

    # Convert (v, w) → wheel angular velocities
    wheel_radius = 0.05
    wheel_base = 0.2  # distance between wheels

    omega_l = (v - (wheel_base / 2) * w) / wheel_radius
    omega_r = (v + (wheel_base / 2) * w) / wheel_radius

    return omega_l, omega_r