module QuadrotorILQR

using LinearAlgebra
using Random

# ==========================
# Quadrotor parameters
# ==========================
const g = 9.81
const L = 0.15
const k_yaw = 0.01
const ILQR_REG = 1e-6

const NX = 12
const NU = 4

# ==========================
# State from physics backend
# ==========================
function get_state_from_bullet(body_id)
    pos, quat = #EDITME
    vel, ang_vel = #EDITME
    roll, pitch, yaw = #EDITME

    return [
        pos[1], pos[2], pos[3],
        vel[1], vel[2], vel[3],
        roll, pitch, yaw,
        ang_vel[1], ang_vel[2], ang_vel[3],
    ]
end

function get_total_mass(body_id)
    total_mass = 0.0
    num_joints = #EDITME

    for link_index in -1:(num_joints - 1)
        dyn = #EDITME
        total_mass += dyn[1]
    end

    return total_mass
end

function keep_gui_alive(dt::Float64 = 1 / 240)
    while #EDITME
        #EDITME
        sleep(dt)
    end
end

# ==========================
# Cost over trajectory
# ==========================
function compute_trajectory_cost(x_history, u_history, x_goal, Q, R, Qf)
    isempty(x_history) && return 0.0

    total_cost = 0.0
    for i in 1:(length(x_history) - 1)
        x = x_history[i]
        u = u_history[i]
        total_cost += cost_stage(x, u, x_goal, Q, R)
    end

    total_cost += cost_final(x_history[end], x_goal, Qf)
    return total_cost
end

# ==========================
# Simulation
# ==========================
function run_simulation(;
    start_pos = [0.0, 0.0, 1.0],
    x_goal_pos = [1.3, 2.1, 1.4],
    connection_mode = #EDITME,
    keep_alive::Bool = false,
    max_steps::Int = 20 * 240,
    log_interval::Int = 240,
    verbose::Bool = true,
    reuse_existing_connection::Bool = false,
)
    physics_client = nothing

    if reuse_existing_connection
        if !#EDITME
            error("reuse_existing_connection=True but no PyBullet connection is active.")
        end
    else
        #physics_client = #EDITME
    end

    try
        # reset world
        #EDITME
        #EDITME
        #EDITME
        #EDITME

        script_dir = @__DIR__
        urdf_path = joinpath(script_dir, "quadrotor", "quadrotor.urdf")
        quad_id = #EDITME

        m = get_total_mass(quad_id)
        inertia_diag = #EDITME

        if verbose
            println("Total mass: ", m)
            println("Base inertia: ", inertia_diag)
        end

        dt = 1 / 240
        N = 180
        replanning_steps = 10

        Q, R, Qf = make_cost_matrices()

        x_goal = zeros(NX)
        x_goal[1:3] .= x_goal_pos

        u_hover = ones(4) .* (m * g / 4.0)
        u_seq = repeat(u_hover .* 1.01, 1, N)'  # N×4

        step = 0
        x_history = Vector{Vector{Float64}}()
        u_history = Vector{Vector{Float64}}()

        if verbose
            println("Simulation started.")
        end

        while step < max_steps
            x = get_state_from_bullet(quad_id)
            push!(x_history, copy(x))

            if step % replanning_steps == 0
                u_seq = ilqr(x, x_goal, N, dt, m, inertia_diag, Q, R, Qf, u_seq)
            end

            u = vec(u_seq[1, :])
            push!(u_history, copy(u))

            apply_motor_forces(quad_id, u)

            u_seq = vcat(u_seq[2:end, :], u_seq[end, :]')

            #EDITME
            if connection_mode == #EDITME
                sleep(dt)
            end

            step += 1

            if verbose && log_interval > 0 && step % log_interval == 0
                println("t=$(step * dt): pos=$(x[1:3]) vel=$(x[4:6]) thrust=$(u)")
            end

            pos_error = norm(x[1:3] .- x_goal[1:3])
            vertical_speed = abs(x[6])

            if pos_error < 0.1 && vertical_speed < 0.15 && step > 300
                if verbose
                    println("Reached goal!")
                end
                break
            end
        end

        if verbose && step >= max_steps
            println("Max simulation time reached.")
        end

        final_state = isempty(x_history) ? zeros(NX) : x_history[end]
        trajectory_total_cost = compute_trajectory_cost(x_history, u_history, x_goal, Q, R, Qf)

        result = Dict(
            "start_x" => start_pos[1],
            "start_y" => start_pos[2],
            "start_z" => start_pos[3],
            "goal_x" => x_goal_pos[1],
            "goal_y" => x_goal_pos[2],
            "goal_z" => x_goal_pos[3],
            "steps" => step,
            "reached_goal" => (norm(final_state[1:3] .- x_goal[1:3]) < 0.1 &&
                               abs(final_state[6]) < 0.15),
            "final_x" => final_state[1],
            "final_y" => final_state[2],
            "final_z" => final_state[3],
            "final_vx" => final_state[4],
            "final_vy" => final_state[5],
            "final_vz" => final_state[6],
            "final_position_error" => norm(final_state[1:3] .- x_goal[1:3]),
            "trajectory_total_cost" => trajectory_total_cost,
        )

        if keep_alive && connection_mode == #EDITME
            println("Total trajectory cost: $(trajectory_total_cost)")
            keep_gui_alive(dt)
        end

        return result

    catch e
        println("Simulation error: ", e)
        if keep_alive && connection_mode == #EDITME
            println("Simulation failed. Keeping the window open.")
            keep_gui_alive()
        end
        rethrow()
    finally
        if physics_client !== nothing #&& #EDITME
            #EDITME
        end
    end
end

# ==========================
# Apply rotor forces
# ==========================
function apply_motor_forces(body_id, u::AbstractVector{<:Real})
    rotor_positions_body = [
        L   0   0;
        0   L   0;
       -L   0   0;
        0  -L   0
    ]
    yaw_signs = [1.0, -1.0, 1.0, -1.0]

    for i in 1:4
        r_body = rotor_positions_body[i, :]
        f = float(u[i])

        #EDITME
    end

    yaw_torque = float(k_yaw * dot(yaw_signs, u))

    #EDITME
end

# ==========================
# Cost
# ==========================
function make_cost_matrices()
    Q = Diagonal([
        8.0, 8.0, 80.0,      # position
        1.0, 1.0, 25.0,      # velocity
        50.0, 50.0, 20.0,    # angles
        5.0, 5.0, 8.0        # angular rates
    ])

    R_cost = Diagonal([0.05, 0.05, 0.05, 0.05])
    Qf = 30 .* Q

    return Q, R_cost, Qf
end

function cost_stage(x, u, x_goal, Q, R)
    dx = x .- x_goal
    return dx' * Q * dx + u' * R * u
end

function cost_final(x, x_goal, Qf)
    dx = x .- x_goal
    return dx' * Qf * dx
end

# ==========================
# iLQR
# ==========================
function build_linearized_dynamics(dt, m, inertia_diag, yaw)
    ix, iy, iz = inertia_diag
    c_yaw = cos(yaw)
    s_yaw = sin(yaw)

    A = Matrix(I, NX, NX)
    A[1, 4] = dt
    A[2, 5] = dt
    A[3, 6] = dt
    A[7,10] = dt
    A[8,11] = dt
    A[9,12] = dt

    A[4, 7] = -g * s_yaw * dt
    A[4, 8] =  g * c_yaw * dt
    A[5, 7] = -g * c_yaw * dt
    A[5, 8] = -g * s_yaw * dt

    B = zeros(NX, NU)
    B[6, :] .= dt / m
    B[10, :] .= dt .* [0.0, L / ix, 0.0, -L / ix]
    B[11, :] .= dt .* [-L / iy, 0.0, L / iy, 0.0]
    B[12, :] .= dt * k_yaw .* [1.0, -1.0, 1.0, -1.0] ./ iz

    return A, B
end

function rollout_dynamics(x0, delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover)
    N = size(delta_u_seq, 1)
    x_seq = zeros(N + 1, NX)
    u_abs_seq = zeros(N, NU)
    x_seq[1, :] .= x0
    total_cost = 0.0

    for k in 1:N
        A, B = build_linearized_dynamics(dt, m, inertia_diag, x_seq[k, 9])
        u_abs = clamp.(hover .+ delta_u_seq[k, :], 0.0, 2.5 * hover)
        delta_u_clipped = u_abs .- hover
        x_seq[k + 1, :] .= A * x_seq[k, :] + B * delta_u_clipped
        u_abs_seq[k, :] .= u_abs
        total_cost += cost_stage(x_seq[k, :], delta_u_clipped, x_goal, Q, R)
    end

    total_cost += cost_final(x_seq[end, :], x_goal, Qf)
    return x_seq, u_abs_seq, total_cost
end

function ilqr(x0, x_goal, N, dt, m, inertia_diag, Q, R, Qf, u_init=nothing; max_iter=25)
    hover = m * g / 4.0

    delta_u_seq =
        u_init === nothing ? zeros(N, NU) :
        clamp.(u_init .- hover, -hover, 1.5 * hover)

    for _ in 1:max_iter
        x_seq, _, current_cost = rollout_dynamics(
            x0, delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover
        )

        Vx = 2 .* Qf * (x_seq[end, :] .- x_goal)
        Vxx = 2 .* Qf

        K = zeros(N, NU, NX)
        k_ff = zeros(N, NU)

        for k in N:-1:1
            A, B = build_linearized_dynamics(dt, m, inertia_diag, x_seq[k, 9])
            dx = x_seq[k, :] .- x_goal

            lx = 2 .* Q * dx
            lu = 2 .* R * delta_u_seq[k, :]
            lxx = 2 .* Q
            luu = 2 .* R
            lux = zeros(NU, NX)

            Qx  = lx + A' * Vx
            Qu  = lu + B' * Vx
            Qxx = lxx + A' * Vxx * A
            Quu = luu + B' * Vxx * B
            Qux = lux + B' * Vxx * A

            Quu_reg = Quu + ILQR_REG .* I
            Quu_inv = inv(Quu_reg)

            K[k, :, :] = -Quu_inv * Qux
            k_ff[k, :] = -Quu_inv * Qu

            Vx = Qx + K[k, :, :]' * Quu * k_ff[k, :] +
                 K[k, :, :]' * Qu + Qux' * k_ff[k, :]
            Vxx = Qxx + K[k, :, :]' * Quu * K[k, :, :] +
                  K[k, :, :]' * Qux + Qux' * K[k, :, :]
            Vxx = 0.5 .* (Vxx + Vxx')
        end

        improved = false
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05)
            trial_delta_u_seq = zeros(size(delta_u_seq))
            trial_x_seq = zeros(size(x_seq))
            trial_x_seq[1, :] .= x0

            for k in 1:N
                dx = trial_x_seq[k, :] .- x_seq[k, :]
                trial_delta_u_seq[k, :] .= delta_u_seq[k, :] .+
                                           alpha .* k_ff[k, :] .+
                                           K[k, :, :] * dx
                A, B = build_linearized_dynamics(dt, m, inertia_diag, trial_x_seq[k, 9])
                u_abs = clamp.(hover .+ trial_delta_u_seq[k, :], 0.0, 2.5 * hover)
                trial_delta_u_seq[k, :] .= u_abs .- hover
                trial_x_seq[k + 1, :] .= A * trial_x_seq[k, :] + B * trial_delta_u_seq[k, :]
            end

            _, trial_u_abs_seq, trial_cost = rollout_dynamics(
                x0, trial_delta_u_seq, x_goal, dt, m, inertia_diag, Q, R, Qf, hover
            )

            if trial_cost < current_cost
                delta_u_seq .= trial_u_abs_seq .- hover
                improved = true
                break
            end
        end

        !improved && break
    end

    return clamp.(hover .+ delta_u_seq, 0.0, 2.5 * hover)
end

# ==========================
# Main
# ==========================
function main()
    result = run_simulation(
        start_pos = [0.0, 0.0, 1.0],
        x_goal_pos = [1.3, 2.1, 1.4],
        connection_mode = #EDITME,
        keep_alive = true,
    )
    println("Total trajectory cost: $(result["trajectory_total_cost"])")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module
