## L1DRAC with a quadrotor iLQR baseline controller
##
## This is a Julia port of the controller/model pieces used by:
##   ME-598-Quadcopter/SanyuWork/IdealSystem.py
##   ME-598-Quadcopter/SanyuWork/NonIdealSystem.py
##
## It keeps L1DRAC's adaptive-control simulation structure, but replaces the
## double-integrator pole-placement baseline with the quadrotor iLQR baseline.

using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using StaticArrays
using Plots
using JLD2
using Random

const NX_QUAD = 12
const NU_QUAD = 4
const GRAVITY = 9.81
const ARM_LENGTH = 0.15
const K_YAW = 0.01
const ILQR_REG = 1e-6
const DEFAULT_INITIAL_MEAN_OFFSETS = Dict{Symbol, Vector{Float64}}(
    :nominal_sys => [-1.5, -1.5, 0.0],
    :true_sys => [-1.5, -1.5, 0.0],
    :L1_sys => [-1.5, -1.5, 0.0],
)
const DEFAULT_WIND_PROFILE = (
    wind_bias_x = 0.15,
    wind_bias_y = -0.20,
    wind_amp_x = 0.0,
    wind_amp_y = 0.0,
    wind_freq_x_hz = 0.18,
    wind_freq_y_hz = 0.15,
    wind_phase_x = 1.4747,
    wind_phase_y = 3.7365,
    gust_start_sec = 2.0,
    gust_duration_sec = 1.0,
    gust_force_x = 0.0,
    gust_force_y = 0.0,
    gust_force_z = 0.0,
)
const NO_WIND_PROFILE = (
    wind_bias_x = 0.0,
    wind_bias_y = 0.0,
    wind_amp_x = 0.0,
    wind_amp_y = 0.0,
    wind_freq_x_hz = 0.18,
    wind_freq_y_hz = 0.15,
    wind_phase_x = 0.0,
    wind_phase_y = 0.0,
    gust_start_sec = 0.0,
    gust_duration_sec = 0.0,
    gust_force_x = 0.0,
    gust_force_y = 0.0,
    gust_force_z = 0.0,
)
const WIND_GUST_PROFILE = (
    wind_bias_x = 1.25,
    wind_bias_y = -1.00,
    wind_amp_x = 0.45,
    wind_amp_y = 0.35,
    wind_freq_x_hz = 0.35,
    wind_freq_y_hz = 0.28,
    wind_phase_x = 1.4747,
    wind_phase_y = 3.7365,
    gust_start_sec = 3.0,
    gust_duration_sec = 1.6,
    gust_force_x = 5.00,
    gust_force_y = -4.00,
    gust_force_z = 1.20,
)
const DEFAULT_FAULT_PROFILE = (
    failure_time = Inf,
    failed_rotor = 1,
    failed_rotor_scale = 1.0,
    sinusoidal_enabled = false,
    sinusoidal_mean_scale = 1.0,
    sinusoidal_amp = 0.0,
    sinusoidal_freq_hz = 0.0,
)
const PROPELLER_FAILURE_PROFILE = (
    failure_time = 2.0,
    failed_rotor = 1,
    failed_rotor_scale = 0.70,
    sinusoidal_enabled = false,
    sinusoidal_mean_scale = 0.70,
    sinusoidal_amp = 0.0,
    sinusoidal_freq_hz = 0.0,
)
const VARIABLE_THRUST_PROFILE = (
    failure_time = 2.0,
    failed_rotor = 1,
    failed_rotor_scale = 1.0,
    sinusoidal_enabled = true,
    sinusoidal_mean_scale = 0.50,
    sinusoidal_amp = 0.35,
    sinusoidal_freq_hz = 2.0,
)
const DEFAULT_AERO_DAMAGE_PROFILE = (
    damage_time = Inf,
    drag_x = 0.0,
    drag_y = 0.0,
    roll_torque_bias = 0.0,
    pitch_torque_bias = 0.0,
    yaw_torque_bias = 0.0,
)
const DAMAGED_AIRFRAME_PROFILE = (
    damage_time = 2.0,
    drag_x = 0.25,
    drag_y = 0.18,
    roll_torque_bias = 0.006,
    pitch_torque_bias = -0.005,
    yaw_torque_bias = 0.002,
)

function make_cost_matrices()
    Q = Diagonal([
        180.0, 180.0, 80.0,    # position
        35.0, 35.0, 25.0,      # velocity
        35.0, 35.0, 20.0,      # angles
        8.0, 8.0, 8.0          # angular rates
    ])
    R = 0.05I(NU_QUAD)
    Qf = 30.0 * Q
    return Matrix(Q), Matrix(R), Matrix(Qf)
end

function build_linearized_dynamics(dt, mass, inertia_diag, yaw)
    ix, iy, iz = inertia_diag[1], inertia_diag[2], inertia_diag[3]          
    c_yaw = cos(yaw)
    s_yaw = sin(yaw)

    A = Matrix{Float64}(I, NX_QUAD, NX_QUAD)
    A[1, 4] = dt
    A[2, 5] = dt
    A[3, 6] = dt
    A[7, 10] = dt
    A[8, 11] = dt
    A[9, 12] = dt

    A[4, 7] = -GRAVITY * s_yaw * dt
    A[4, 8] =  GRAVITY * c_yaw * dt
    A[5, 7] = -GRAVITY * c_yaw * dt
    A[5, 8] = -GRAVITY * s_yaw * dt

    B = zeros(NX_QUAD, NU_QUAD)
    B[6, :] .= dt / mass
    B[10, :] .= dt .* [0.0, ARM_LENGTH / ix, 0.0, -ARM_LENGTH / ix]
    B[11, :] .= dt .* [-ARM_LENGTH / iy, 0.0, ARM_LENGTH / iy, 0.0]
    B[12, :] .= dt .* K_YAW .* [1.0, -1.0, 1.0, -1.0] ./ iz

    return A, B
end

function continuous_input_matrix(mass, inertia_diag)
    _, Bdt = build_linearized_dynamics(1.0, mass, inertia_diag, 0.0)
    return Bdt
end

function controllability_matrix(A, B)
    blocks = Matrix{Float64}[]
    Ak = Matrix{Float64}(I, size(A, 1), size(A, 1))

    for _ in 1:size(A, 1)
        push!(blocks, Ak * B)
        Ak = Ak * A
    end

    return hcat(blocks...)
end

function quadrotor_controllability_rank(; dt=1 / 240,
                                        mass=1.0,
                                        inertia_diag=[0.005, 0.005, 0.009],
                                        yaw=0.0,
                                        actuator_scale=ones(NU_QUAD),
                                        atol=1e-8)
    A, B = build_linearized_dynamics(dt, mass, inertia_diag, yaw)
    B_failed = B * Diagonal(actuator_scale)
    C = controllability_matrix(A, B_failed)
    return (
        rank = rank(C; atol=atol),
        full_rank = rank(C; atol=atol) == NX_QUAD,
        singular_values = svdvals(C),
    )
end

function check_failed_rotor_controllability(; failed_rotor=1,
                                            failed_rotor_scale=0.70,
                                            yaw=0.0,
                                            atol=1e-8)
    actuator_scale = ones(NU_QUAD)
    actuator_scale[failed_rotor] = failed_rotor_scale
    result = quadrotor_controllability_rank(;
        actuator_scale=actuator_scale,
        yaw=yaw,
        atol=atol,
    )
    @info "Failed rotor controllability" failed_rotor failed_rotor_scale yaw result.rank result.full_rank minimum_singular_value=minimum(result.singular_values)
    return result
end

function cost_stage(x, delta_u, x_goal, Q, R)
    dx = x - x_goal
    return dot(dx, Q * dx) + dot(delta_u, R * delta_u)
end

function cost_final(x, x_goal, Qf)
    dx = x - x_goal
    return dot(dx, Qf * dx)
end

function obstacle_barrier_terms(x, obstacles, clearance;
                                weight=0.0, eps=1e-3, huge_cost=1e12)
    cost = 0.0
    grad = zeros(NX_QUAD)
    hess = zeros(NX_QUAD, NX_QUAD)

    (weight <= 0.0 || isempty(obstacles)) && return cost, grad, hess

    pos = x[1:3]
    for obs in obstacles
        center = obs.center
        inflated_radius = obs.radius + clearance
        rvec = pos .- center
        dist = max(norm(rvec), eps)
        gap = dist - inflated_radius

        if gap <= eps
            cost += huge_cost + weight * (eps - gap)^2 / eps^2
            direction = rvec ./ dist
            grad[1:3] .+= -2.0 * weight * (eps - gap) / eps^2 .* direction
            hess[1:3, 1:3] .+= (2.0 * weight / eps^2) .* (direction * direction')
            continue
        end

        direction = rvec ./ dist
        cost += -weight * log(gap)
        grad[1:3] .+= -(weight / gap) .* direction
        hess_pos = (weight / gap^2) .* (direction * direction') -
                   (weight / gap) .* ((Matrix{Float64}(I, 3, 3) - direction * direction') ./ dist)
        hess[1:3, 1:3] .+= 0.5 .* (hess_pos + hess_pos')
    end

    return cost, grad, hess
end

function cost_stage(x, delta_u, x_goal, Q, R, obstacles, clearance, barrier_weight)
    base_cost = cost_stage(x, delta_u, x_goal, Q, R)
    barrier_cost, _, _ = obstacle_barrier_terms(x, obstacles, clearance; weight=barrier_weight)
    return base_cost + barrier_cost
end

function cost_final(x, x_goal, Qf, obstacles, clearance, barrier_weight)
    base_cost = cost_final(x, x_goal, Qf)
    barrier_cost, _, _ = obstacle_barrier_terms(x, obstacles, clearance; weight=barrier_weight)
    return base_cost + barrier_cost
end

function rollout_dynamics(x0, delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover;
                          obstacles=(), obstacle_clearance=0.0, obstacle_barrier_weight=0.0)
    horizon = size(delta_u_seq, 1)
    x_seq = zeros(horizon + 1, NX_QUAD)
    u_abs_seq = zeros(horizon, NU_QUAD)
    x_seq[1, :] .= x0
    total_cost = 0.0

    for k in 1:horizon
        A, B = build_linearized_dynamics(dt, mass, inertia_diag, x_seq[k, 9])
        u_abs = clamp.(hover .+ delta_u_seq[k, :], 0.0, 2.5 * hover)
        delta_u_clipped = u_abs .- hover
        x_seq[k + 1, :] .= A * x_seq[k, :] .+ B * delta_u_clipped
        u_abs_seq[k, :] .= u_abs
        total_cost += cost_stage(x_seq[k, :], delta_u_clipped, x_goal, Q, R,
                                 obstacles, obstacle_clearance, obstacle_barrier_weight)
    end

    total_cost += cost_final(x_seq[end, :], x_goal, Qf,
                             obstacles, obstacle_clearance, obstacle_barrier_weight)
    return x_seq, u_abs_seq, total_cost
end

function ilqr(x0, x_goal, horizon, dt, mass, inertia_diag, Q, R, Qf;
              u_init=nothing, max_iter=5,
              obstacles=(), obstacle_clearance=0.0, obstacle_barrier_weight=0.0)
    hover = mass * GRAVITY / 4.0
    if u_init === nothing
        delta_u_seq = zeros(horizon, NU_QUAD)
    else
        delta_u_seq = clamp.(u_init .- hover, -hover, 1.5 * hover)
    end

    for _ in 1:max_iter
        x_seq, _, current_cost = rollout_dynamics(
            x0, delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover;
            obstacles=obstacles,
            obstacle_clearance=obstacle_clearance,
            obstacle_barrier_weight=obstacle_barrier_weight,
        )

        Vx = 2.0 .* (Qf * (x_seq[end, :] - x_goal))
        Vxx = 2.0 .* Qf
        _, final_barrier_grad, final_barrier_hess = obstacle_barrier_terms(
            x_seq[end, :], obstacles, obstacle_clearance; weight=obstacle_barrier_weight
        )
        Vx .+= final_barrier_grad
        Vxx .+= final_barrier_hess
        K = zeros(horizon, NU_QUAD, NX_QUAD)
        k_ff = zeros(horizon, NU_QUAD)

        for k in horizon:-1:1
            A, B = build_linearized_dynamics(dt, mass, inertia_diag, x_seq[k, 9])
            dx = x_seq[k, :] - x_goal

            lx = 2.0 .* (Q * dx)
            lu = 2.0 .* (R * delta_u_seq[k, :])
            lxx = 2.0 .* Q
            luu = 2.0 .* R
            lux = zeros(NU_QUAD, NX_QUAD)
            _, barrier_grad, barrier_hess = obstacle_barrier_terms(
                x_seq[k, :], obstacles, obstacle_clearance; weight=obstacle_barrier_weight
            )
            lx .+= barrier_grad
            lxx .+= barrier_hess

            Qx = lx + A' * Vx
            Qu = lu + B' * Vx
            Qxx = lxx + A' * Vxx * A
            Quu = luu + B' * Vxx * B
            Qux = lux + B' * Vxx * A

            Quu_reg = Quu + ILQR_REG * I(NU_QUAD)
            Quu_inv = inv(Quu_reg)

            K[k, :, :] .= -Quu_inv * Qux
            k_ff[k, :] .= -Quu_inv * Qu

            Kk = K[k, :, :]
            kk = k_ff[k, :]
            Vx = Qx + Kk' * Quu * kk + Kk' * Qu + Qux' * kk
            Vxx = Qxx + Kk' * Quu * Kk + Kk' * Qux + Qux' * Kk
            Vxx = 0.5 .* (Vxx + Vxx')
        end

        improved = false
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05)
            trial_delta_u_seq = zeros(size(delta_u_seq))
            trial_x_seq = zeros(size(x_seq))
            trial_x_seq[1, :] .= x0

            for k in 1:horizon
                dx = trial_x_seq[k, :] - x_seq[k, :]
                trial_delta_u_seq[k, :] .= delta_u_seq[k, :] .+
                                            alpha .* k_ff[k, :] .+
                                            K[k, :, :] * dx
                A, B = build_linearized_dynamics(dt, mass, inertia_diag, trial_x_seq[k, 9])
                u_abs = clamp.(hover .+ trial_delta_u_seq[k, :], 0.0, 2.5 * hover)
                trial_delta_u_seq[k, :] .= u_abs .- hover
                trial_x_seq[k + 1, :] .= A * trial_x_seq[k, :] .+ B * trial_delta_u_seq[k, :]
            end

            _, trial_u_abs_seq, trial_cost = rollout_dynamics(
                x0, trial_delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover;
                obstacles=obstacles,
                obstacle_clearance=obstacle_clearance,
                obstacle_barrier_weight=obstacle_barrier_weight,
            )

            if trial_cost < current_cost
                delta_u_seq .= trial_u_abs_seq .- hover
                improved = true
                break
            end
        end

        improved || break
    end

    return clamp.(hover .+ delta_u_seq, 0.0, 2.5 * hover)
end

function dlqr_gain(A, B, Q, R; max_iter=500, tol=1e-9)
    P = copy(Q)

    for _ in 1:max_iter
        P_next = Q + A' * P * A -
                 A' * P * B * ((R + B' * P * B) \ (B' * P * A))
        if norm(P_next - P) < tol
            P = P_next
            break
        end
        P = P_next
    end

    return (R + B' * P * B) \ (B' * P * A)
end

function quadrotor_reference_state(t, dp)
    if hasproperty(dp, :reference_func) && dp.reference_func !== nothing
        return dp.reference_func(t)
    end
    return dp.x_goal
end

function quadrotor_ilqr_input(t, x, dp)
    x_ref = quadrotor_reference_state(t + dp.reference_lookahead, dp)
    u_seq = ilqr(
        collect(x),
        x_ref,
        dp.horizon,
        dp.ilqr_dt,
        dp.mass_nominal,
        dp.inertia_nominal,
        dp.Q,
        dp.R,
        dp.Qf;
        u_init=dp.u_hover_init,
        max_iter=dp.ilqr_max_iter,
        obstacles=dp.obstacles,
        obstacle_clearance=dp.obstacle_clearance,
        obstacle_barrier_weight=dp.obstacle_barrier_weight,
    )
    return u_seq[1, :]
end

function quadrotor_lqr_input(t, x, dp)
    x_ref = quadrotor_reference_state(t + dp.reference_lookahead, dp)
    delta_u = -dp.K_lqr * (collect(x) - x_ref)
    return clamp.(dp.hover_nominal .+ delta_u, 0.0, 2.5 * dp.hover_nominal)
end

function quadrotor_control_input(t, x, dp)
    dp.controller_type == :lqr && return quadrotor_lqr_input(t, x, dp)
    return quadrotor_ilqr_input(t, x, dp)
end

function quadrotor_nominal_drift(t, x, dp)
    yaw = x[9]
    _, Bdt = build_linearized_dynamics(1.0, dp.mass_nominal, dp.inertia_nominal, yaw)
    u = quadrotor_control_input(t, x, dp)
    delta_u = u .- dp.hover_nominal
    dx = Bdt * delta_u

    dx[1] += x[4]
    dx[2] += x[5]
    dx[3] += x[6]
    dx[4] += -GRAVITY * sin(yaw) * x[7] + GRAVITY * cos(yaw) * x[8]
    dx[5] += -GRAVITY * cos(yaw) * x[7] - GRAVITY * sin(yaw) * x[8]
    dx[7] += x[10]
    dx[8] += x[11]
    dx[9] += x[12]

    return dx
end

function quadrotor_scaled_plant_uncertainty(t, x, dp)
    yaw = x[9]
    _, B_nominal = build_linearized_dynamics(1.0, dp.mass_nominal, dp.inertia_nominal, yaw)
    _, B_true = build_linearized_dynamics(1.0, dp.mass_true, dp.inertia_true, yaw)
    B_true[6, :] .*= dp.plant_thrust_scale
    B_true[12, :] .*= dp.plant_yaw_coeff_scale

    actuator_scale = ones(NU_QUAD)
    if t >= dp.fault_profile.failure_time
        failed_rotor = dp.fault_profile.failed_rotor
        if dp.fault_profile.sinusoidal_enabled
            tau = t - dp.fault_profile.failure_time
            rotor_scale = dp.fault_profile.sinusoidal_mean_scale +
                          dp.fault_profile.sinusoidal_amp *
                          sin(2.0 * pi * dp.fault_profile.sinusoidal_freq_hz * tau)
            actuator_scale[failed_rotor] = clamp(rotor_scale, 0.05, 1.2)
        else
            actuator_scale[failed_rotor] = dp.fault_profile.failed_rotor_scale
        end
    end

    u = quadrotor_control_input(t, x, dp)
    nominal_delta_u = u .- dp.hover_nominal
    true_delta_u = actuator_scale .* u .- dp.hover_nominal
    return B_true * true_delta_u - B_nominal * nominal_delta_u
end

function wind_force_world(t, profile)
    wind_fx = profile.wind_bias_x +
              profile.wind_amp_x * sin(2.0 * pi * profile.wind_freq_x_hz * t + profile.wind_phase_x)
    wind_fy = profile.wind_bias_y +
              profile.wind_amp_y * sin(2.0 * pi * profile.wind_freq_y_hz * t + profile.wind_phase_y)
    wind_fz = 0.0

    gust_stop_sec = profile.gust_start_sec + profile.gust_duration_sec
    if profile.gust_start_sec <= t <= gust_stop_sec && profile.gust_duration_sec > 0.0
        gust_phase = (t - profile.gust_start_sec) / profile.gust_duration_sec
        gust_envelope = sin(pi * gust_phase)
        wind_fx += profile.gust_force_x * gust_envelope
        wind_fy += profile.gust_force_y * gust_envelope
        wind_fz += profile.gust_force_z * gust_envelope
    end

    return [wind_fx, wind_fy, wind_fz]
end

function quadrotor_wind_disturbance(t, x, dp)
    dx = zeros(NX_QUAD)
    wind_force = wind_force_world(t, dp.wind_profile)
    dx[4:6] .= wind_force ./ dp.mass_true
    return dx
end

function quadrotor_aero_damage_disturbance(t, x, dp)
    dx = zeros(NX_QUAD)
    t < dp.aero_damage_profile.damage_time && return dx

    dx[4] -= dp.aero_damage_profile.drag_x * x[4] * abs(x[4]) / dp.mass_true
    dx[5] -= dp.aero_damage_profile.drag_y * x[5] * abs(x[5]) / dp.mass_true
    dx[10] += dp.aero_damage_profile.roll_torque_bias / dp.inertia_true[1]
    dx[11] += dp.aero_damage_profile.pitch_torque_bias / dp.inertia_true[2]
    dx[12] += dp.aero_damage_profile.yaw_torque_bias / dp.inertia_true[3]
    return dx
end

function quadrotor_lumped_drift_uncertainty(t, x, dp)
    plant_uncertainty = dp.plant_mismatch_enabled ?
        quadrotor_scaled_plant_uncertainty(t, x, dp) :
        zeros(NX_QUAD)

    return plant_uncertainty +
           quadrotor_wind_disturbance(t, x, dp) +
           quadrotor_aero_damage_disturbance(t, x, dp)
end

function quadrotor_l1_control_map(t, x, lambda_hat, dp)
    G = dp.B_nominal
    corrected_lambda = copy(lambda_hat)
    x_ref = quadrotor_reference_state(t, dp)

    lateral_accel = lambda_hat[4:5] +
                    dp.L1_lateral_position_gain .* (x[1:2] .- x_ref[1:2]) +
                    dp.L1_lateral_velocity_gain .* (x[4:5] .- x_ref[4:5])
    if norm(lateral_accel) > 1e-9
        yaw = x[9]
        lateral_tilt_map = [
            -GRAVITY * sin(yaw)   GRAVITY * cos(yaw);
            -GRAVITY * cos(yaw)  -GRAVITY * sin(yaw)
        ]
        target_tilt = lateral_tilt_map \ (-lateral_accel)
        target_tilt = clamp.(target_tilt, -dp.L1_max_tilt_correction, dp.L1_max_tilt_correction)

        desired_roll_accel = dp.L1_lateral_tilt_gain * (target_tilt[1] - x[7]) -
                             dp.L1_lateral_rate_damping * x[10]
        desired_pitch_accel = dp.L1_lateral_tilt_gain * (target_tilt[2] - x[8]) -
                              dp.L1_lateral_rate_damping * x[11]

        corrected_lambda[10] -= desired_roll_accel
        corrected_lambda[11] -= desired_pitch_accel
    end

    weights = zeros(NX_QUAD)
    weights[6] = 2.0
    weights[10] = 1.8
    weights[11] = 1.8
    weights[12] = 1.0

    W = Diagonal(weights)
    mapped = (G' * W * G + dp.L1_control_reg * I(NU_QUAD)) \ (G' * W * corrected_lambda)
    return clamp.(mapped, -dp.L1_control_limit, dp.L1_control_limit)
end

function scenario_title(output_tag)
    tag = Symbol(output_tag)
    tag == :baseline && return "Baseline: Brownian and aleatoric uncertainty only"
    tag == :propeller_failure && return "Propeller Failure: rotor 1 at 70% after 2.0 s"
    tag == :variable_thrust && return "Variable Thrust: rotor 1 sinusoidal effectiveness after 2.0 s"
    tag == :wind_gust && return "Strong Wind Gust: steady wind plus severe smooth gust"
    tag == :aero_damage && return "Aero Damage: drag and torque bias after 2.0 s"
    tag == :combined_failure && return "Combined Failure: rotor loss, gust, and aero damage"

    words = split(replace(string(output_tag), "_" => " "))
    return join(uppercasefirst.(words), " ")
end

function _initial_offset_vector(offset, n)
    offset_vec = zeros(n)
    ncopy = min(length(offset), n)
    offset_vec[1:ncopy] .= offset[1:ncopy]
    return offset_vec
end

function setup_system(; Ntraj=2, t_final=1.0, dt=1e-2, save_stride=5,
                      ilqr_horizon=80, ilqr_max_iter=5,
                      controller_type=:ilqr,
                      brownian_sigma=0.15, aleatoric_sigma=0.15,
                      L1_omega=10.0, L1_sample_steps=25,
                      L1_predictor_lambda=25.0,
                      L1_control_reg=1e-3, L1_control_limit=2.0,
                      L1_lateral_tilt_gain=8.0,
                      L1_lateral_rate_damping=2.5,
                      L1_max_tilt_correction=0.45,
                      L1_lateral_position_gain=0.0,
                      L1_lateral_velocity_gain=0.0,
                      reference_func=nothing,
                      reference_lookahead=0.0,
                      obstacles=(),
                      obstacle_clearance=0.0,
                      obstacle_barrier_weight=0.0,
                      initial_mean_offset=zeros(NX_QUAD),
                      simulated_system=nothing,
                      wind_profile=DEFAULT_WIND_PROFILE,
                      fault_profile=DEFAULT_FAULT_PROFILE,
                      aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
                      plant_mismatch_enabled=true)
    tspan = (0.0, t_final)
    saveat = save_stride * dt
    simulation_parameters = sim_params(tspan, dt, Ntraj, saveat)

    n, m, d = NX_QUAD, NU_QUAD, NU_QUAD
    system_dimensions = sys_dims(n, m, d)

    mass_nominal = 1.0
    inertia_nominal = [0.005, 0.005, 0.009]
    plant_mass_scale = 1.30
    plant_inertia_scale = 1.15
    plant_thrust_scale = 0.60
    plant_yaw_coeff_scale = 1.30

    Q, R, Qf = make_cost_matrices()
    x_goal = zeros(NX_QUAD)
    x_goal[1:3] .= [1.3, 2.1, 1.4]
    hover_nominal = mass_nominal * GRAVITY / 4.0
    A_lqr, B_lqr = build_linearized_dynamics(1 / 240, mass_nominal, inertia_nominal, 0.0)
    K_lqr = dlqr_gain(A_lqr, B_lqr, Q, R)

    dp = (
        controller_type = controller_type,
        mass_nominal = mass_nominal,
        inertia_nominal = inertia_nominal,
        mass_true = mass_nominal * plant_mass_scale,
        inertia_true = inertia_nominal .* plant_inertia_scale,
        plant_thrust_scale = plant_thrust_scale,
        plant_yaw_coeff_scale = plant_yaw_coeff_scale,
        Q = Q,
        R = R,
        Qf = Qf,
        x_goal = x_goal,
        reference_func = reference_func,
        reference_lookahead = reference_lookahead,
        obstacles = obstacles,
        obstacle_clearance = obstacle_clearance,
        obstacle_barrier_weight = obstacle_barrier_weight,
        hover_nominal = hover_nominal,
        u_hover_init = fill(1.01 * hover_nominal, ilqr_horizon, NU_QUAD),
        horizon = ilqr_horizon,
        ilqr_dt = 1 / 240,
        ilqr_max_iter = ilqr_max_iter,
        K_lqr = K_lqr,
        B_nominal = continuous_input_matrix(mass_nominal, inertia_nominal),
        wind_profile = wind_profile,
        fault_profile = fault_profile,
        aero_damage_profile = aero_damage_profile,
        plant_mismatch_enabled = plant_mismatch_enabled,
        L1_control_reg = L1_control_reg,
        L1_control_limit = L1_control_limit,
        L1_lateral_tilt_gain = L1_lateral_tilt_gain,
        L1_lateral_rate_damping = L1_lateral_rate_damping,
        L1_max_tilt_correction = L1_max_tilt_correction,
        L1_lateral_position_gain = L1_lateral_position_gain,
        L1_lateral_velocity_gain = L1_lateral_velocity_gain,
        l1_control_map = quadrotor_l1_control_map,
    )

    f(t, x, dp) = quadrotor_nominal_drift(t, x, dp)
    g(t, x, dp) = dp.B_nominal
    g_perp(t, x, dp) = Matrix{Float64}(I, n, n)[:, [1, 2, 3, 4, 5, 7, 8, 9]]

    brownian_diffusion = zeros(n, d)
    brownian_diffusion[4, 1] = brownian_sigma
    brownian_diffusion[5, 2] = brownian_sigma
    brownian_diffusion[6, 3] = 0.5 * brownian_sigma
    brownian_diffusion[12, 4] = 0.25 * brownian_sigma

    aleatoric_diffusion = zeros(n, d)
    aleatoric_diffusion[4, 1] = aleatoric_sigma
    aleatoric_diffusion[5, 2] = aleatoric_sigma
    aleatoric_diffusion[6, 3] = 0.5 * aleatoric_sigma
    aleatoric_diffusion[10, 1] = 0.2 * aleatoric_sigma
    aleatoric_diffusion[11, 2] = 0.2 * aleatoric_sigma
    aleatoric_diffusion[12, 4] = 0.25 * aleatoric_sigma

    nominal_diffusion = simulated_system == :nominal_sys ?
        zeros(n, d) :
        brownian_diffusion

    p(t, x, dp) = nominal_diffusion
    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    Lambda_mu(t, x, dp) = quadrotor_lumped_drift_uncertainty(t, x, dp)
    Lambda_sigma(t, x, dp) = aleatoric_diffusion
    uncertain_components = uncertain_vector_fields(Lambda_mu, Lambda_sigma)

    nominal_mu = zeros(n)
    nominal_mu[1:3] .= [0.0, 0.0, 1.0]
    nominal_mu .+= _initial_offset_vector(initial_mean_offset, n)
    true_mu = copy(nominal_mu)
    initial_cov = Diagonal(vcat(fill(0.05, 3), fill(0.02, 3), fill(0.01, 6)))
    initial_distributions = init_dist(
        MvNormal(nominal_mu, initial_cov),
        MvNormal(true_mu, initial_cov),
    )

    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    omega = L1_omega
    sample_time = L1_sample_steps * dt
    predictor_lambda = L1_predictor_lambda
    L1params = drac_params(omega, sample_time, predictor_lambda)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params,
        system_dimensions = system_dimensions
    )
end

function _run_simulations_maybe_quiet(setup; max_GPUs, systems, quiet)
    quiet || return run_simulations(setup; max_GPUs=max_GPUs, systems=systems)

    redirect_stdout(devnull) do
        redirect_stderr(devnull) do
            return run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
        end
    end
end

function _offset_for_system(initial_mean_offsets, system)
    initial_mean_offsets === nothing && return zeros(NX_QUAD)
    return get(initial_mean_offsets, system, zeros(NX_QUAD))
end

function _setup_with_offset(system; initial_mean_offsets, kwargs...)
    return setup_system(;
        kwargs...,
        initial_mean_offset=_offset_for_system(initial_mean_offsets, system),
        simulated_system=system,
    )
end

function _run_spaced_simulations(; systems, max_GPUs, quiet, initial_mean_offsets, kwargs...)
    setup_for_logging = nothing
    results = Dict{Symbol, Any}()

    for system in systems
        setup = _setup_with_offset(system; initial_mean_offsets=initial_mean_offsets, kwargs...)
        setup_for_logging === nothing && (setup_for_logging = setup)
        solutions = _run_simulations_maybe_quiet(setup; max_GPUs=max_GPUs, systems=[system], quiet=quiet)

        system == :nominal_sys && (results[:nominal_sol] = solutions.nominal_sol)
        system == :true_sys && (results[:true_sol] = solutions.true_sol)
        system == :L1_sys && (results[:L1_sol] = solutions.L1_sol)
    end

    return setup_for_logging, (
        nominal_sol = get(results, :nominal_sol, nothing),
        true_sol = get(results, :true_sol, nothing),
        L1_sol = get(results, :L1_sol, nothing),
    )
end

function main(; Ntraj=2, max_GPUs=0, systems=[:nominal_sys, :true_sys, :L1_sys],
              t_final=1.0, dt=1e-2, save_stride=5,
              ilqr_horizon=80, ilqr_max_iter=5,
              controller_type=:ilqr,
              brownian_sigma=0.15, aleatoric_sigma=0.15,
              L1_omega=10.0, L1_sample_steps=25,
              L1_predictor_lambda=25.0,
              L1_control_reg=1e-3, L1_control_limit=2.0,
              L1_lateral_tilt_gain=8.0,
              L1_lateral_rate_damping=2.5,
              L1_max_tilt_correction=0.45,
              L1_lateral_position_gain=0.0,
              L1_lateral_velocity_gain=0.0,
              reference_func=nothing,
              reference_lookahead=0.0,
              obstacles=(),
              obstacle_clearance=0.0,
              obstacle_barrier_weight=0.0,
              initial_mean_offsets=DEFAULT_INITIAL_MEAN_OFFSETS,
              wind_profile=DEFAULT_WIND_PROFILE,
              fault_profile=DEFAULT_FAULT_PROFILE,
              aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
              plant_mismatch_enabled=true,
              quiet=true)
    quiet || @info "Warmup run for JIT compilation"
    _run_spaced_simulations(;
        systems=systems,
        max_GPUs=max_GPUs,
        quiet=quiet,
        initial_mean_offsets=initial_mean_offsets,
        Ntraj=1,
        t_final=min(t_final, 0.1),
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        controller_type=controller_type,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_reg=L1_control_reg,
        L1_control_limit=L1_control_limit,
        L1_lateral_tilt_gain=L1_lateral_tilt_gain,
        L1_lateral_rate_damping=L1_lateral_rate_damping,
        L1_max_tilt_correction=L1_max_tilt_correction,
        L1_lateral_position_gain=L1_lateral_position_gain,
        L1_lateral_velocity_gain=L1_lateral_velocity_gain,
        reference_func=reference_func,
        reference_lookahead=reference_lookahead,
        obstacles=obstacles,
        obstacle_clearance=obstacle_clearance,
        obstacle_barrier_weight=obstacle_barrier_weight,
        wind_profile=wind_profile,
        fault_profile=fault_profile,
        aero_damage_profile=aero_damage_profile,
        plant_mismatch_enabled=plant_mismatch_enabled,
    );

    quiet || @info "Complete run" Ntraj=Ntraj t_final=t_final dt=dt controller_type=controller_type ilqr_horizon=ilqr_horizon ilqr_max_iter=ilqr_max_iter
    setup, solutions = _run_spaced_simulations(;
        systems=systems,
        max_GPUs=max_GPUs,
        quiet=quiet,
        initial_mean_offsets=initial_mean_offsets,
        Ntraj=Ntraj,
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        controller_type=controller_type,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_reg=L1_control_reg,
        L1_control_limit=L1_control_limit,
        L1_lateral_tilt_gain=L1_lateral_tilt_gain,
        L1_lateral_rate_damping=L1_lateral_rate_damping,
        L1_max_tilt_correction=L1_max_tilt_correction,
        L1_lateral_position_gain=L1_lateral_position_gain,
        L1_lateral_velocity_gain=L1_lateral_velocity_gain,
        reference_func=reference_func,
        reference_lookahead=reference_lookahead,
        obstacles=obstacles,
        obstacle_clearance=obstacle_clearance,
        obstacle_barrier_weight=obstacle_barrier_weight,
        wind_profile=wind_profile,
        fault_profile=fault_profile,
        aero_damage_profile=aero_damage_profile,
        plant_mismatch_enabled=plant_mismatch_enabled,
    )
    return setup, solutions
end

function run_long_ensemble(; Ntraj=1000, t_final=5.0, max_GPUs=0,
                           dt=2e-2, save_stride=5,
                           ilqr_horizon=80, ilqr_max_iter=5,
                           controller_type=:ilqr,
                           brownian_sigma=0.15, aleatoric_sigma=0.15,
                           L1_omega=10.0, L1_sample_steps=25,
                           L1_predictor_lambda=25.0,
                           L1_control_reg=1e-3, L1_control_limit=2.0,
                           L1_lateral_tilt_gain=8.0,
                           L1_lateral_rate_damping=2.5,
                           L1_max_tilt_correction=0.45,
                           L1_lateral_position_gain=0.0,
                           L1_lateral_velocity_gain=0.0,
                           reference_func=nothing,
                           reference_lookahead=0.0,
                           wind_profile=DEFAULT_WIND_PROFILE,
                           fault_profile=DEFAULT_FAULT_PROFILE,
                           aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
                           plant_mismatch_enabled=true,
                           quiet=true)
    setup, solutions = main(;
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        controller_type=controller_type,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_reg=L1_control_reg,
        L1_control_limit=L1_control_limit,
        L1_lateral_tilt_gain=L1_lateral_tilt_gain,
        L1_lateral_rate_damping=L1_lateral_rate_damping,
        L1_max_tilt_correction=L1_max_tilt_correction,
        L1_lateral_position_gain=L1_lateral_position_gain,
        L1_lateral_velocity_gain=L1_lateral_velocity_gain,
        reference_func=reference_func,
        reference_lookahead=reference_lookahead,
        wind_profile=wind_profile,
        fault_profile=fault_profile,
        aero_damage_profile=aero_damage_profile,
        plant_mismatch_enabled=plant_mismatch_enabled,
        quiet=quiet,
    )
    log_state_results(setup, solutions)
    return generate_adaptive_comparison_plots()
end

function run_nominal_ensemble_and_plot(; Ntraj=50, t_final=10.0, max_GPUs=0,
                                       dt=1e-2, save_stride=5,
                                       ilqr_horizon=80, ilqr_max_iter=5,
                                       controller_type=:ilqr,
                                       quiet=true)
    setup, solutions = main(;
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        systems=[:nominal_sys],
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        controller_type=controller_type,
        quiet=quiet,
    )
    log_state_results(setup, solutions)
    return generate_nominal_position_plots()
end

function run_uncertain_ensemble_and_plot(; Ntraj=50, t_final=10.0, max_GPUs=0,
                                        dt=1e-2, save_stride=5,
                                        ilqr_horizon=80, ilqr_max_iter=5,
                                        controller_type=:ilqr,
                                        brownian_sigma=0.15,
                                        aleatoric_sigma=0.15,
                                        L1_omega=10.0,
                                        L1_sample_steps=25,
                                        L1_predictor_lambda=25.0,
                                        L1_control_reg=1e-3,
                                        L1_control_limit=2.0,
                                        L1_lateral_tilt_gain=8.0,
                                        L1_lateral_rate_damping=2.5,
                                        L1_max_tilt_correction=0.45,
                                        L1_lateral_position_gain=0.0,
                                        L1_lateral_velocity_gain=0.0,
                                        reference_func=nothing,
                                        reference_lookahead=0.0,
                                        wind_profile=DEFAULT_WIND_PROFILE,
                                        fault_profile=DEFAULT_FAULT_PROFILE,
                                        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
                                        plant_mismatch_enabled=true,
                                        output_tag="default",
                                        quiet=true)
    setup, solutions = main(;
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        systems=[:nominal_sys, :true_sys, :L1_sys],
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        controller_type=controller_type,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_reg=L1_control_reg,
        L1_control_limit=L1_control_limit,
        L1_lateral_tilt_gain=L1_lateral_tilt_gain,
        L1_lateral_rate_damping=L1_lateral_rate_damping,
        L1_max_tilt_correction=L1_max_tilt_correction,
        L1_lateral_position_gain=L1_lateral_position_gain,
        L1_lateral_velocity_gain=L1_lateral_velocity_gain,
        reference_func=reference_func,
        reference_lookahead=reference_lookahead,
        wind_profile=wind_profile,
        fault_profile=fault_profile,
        aero_damage_profile=aero_damage_profile,
        plant_mismatch_enabled=plant_mismatch_enabled,
        quiet=quiet,
    )
    log_path = joinpath(@__DIR__, "quad_ilqr_sol_logs", string(output_tag))
    log_state_results(setup, solutions; path=log_path)
    return generate_three_system_comparison_plots(;
        path=log_path,
        output_prefix="quad_ilqr_$(output_tag)",
        plot_title="Nominal vs True Disturbed vs L1-DRAC: $(scenario_title(output_tag))",
    )
end

function run_no_wind_ensemble_and_plot(; output_tag="no_wind", kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        output_tag=output_tag,
    )
end

function run_propeller_failure_ensemble_and_plot(; output_tag="propeller_failure", kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=PROPELLER_FAILURE_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        output_tag=output_tag,
    )
end

function run_variable_thrust_ensemble_and_plot(; output_tag="variable_thrust",
                                               L1_omega=50.0,
                                               L1_sample_steps=2,
                                               L1_predictor_lambda=100.0,
                                               L1_control_limit=8.0,
                                               kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_limit=L1_control_limit,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=VARIABLE_THRUST_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        output_tag=output_tag,
    )
end

function run_wind_gust_ensemble_and_plot(; output_tag="wind_gust",
                                         ilqr_horizon=40,
                                         L1_omega=60.0,
                                         L1_sample_steps=2,
                                         L1_predictor_lambda=120.0,
                                         L1_control_limit=8.0,
                                         L1_lateral_tilt_gain=45.0,
                                         L1_lateral_rate_damping=10.0,
                                         L1_max_tilt_correction=1.10,
                                         L1_lateral_position_gain=2.50,
                                         L1_lateral_velocity_gain=3.50,
                                         kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        ilqr_horizon=ilqr_horizon,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_limit=L1_control_limit,
        L1_lateral_tilt_gain=L1_lateral_tilt_gain,
        L1_lateral_rate_damping=L1_lateral_rate_damping,
        L1_max_tilt_correction=L1_max_tilt_correction,
        L1_lateral_position_gain=L1_lateral_position_gain,
        L1_lateral_velocity_gain=L1_lateral_velocity_gain,
        wind_profile=WIND_GUST_PROFILE,
        fault_profile=DEFAULT_FAULT_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        output_tag=output_tag,
    )
end

function run_aero_damage_ensemble_and_plot(; output_tag="aero_damage", kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=DEFAULT_FAULT_PROFILE,
        aero_damage_profile=DAMAGED_AIRFRAME_PROFILE,
        output_tag=output_tag,
    )
end

function run_combined_failure_ensemble_and_plot(; output_tag="combined_failure", kwargs...)
    return run_uncertain_ensemble_and_plot(;
        kwargs...,
        wind_profile=WIND_GUST_PROFILE,
        fault_profile=PROPELLER_FAILURE_PROFILE,
        aero_damage_profile=DAMAGED_AIRFRAME_PROFILE,
        output_tag=output_tag,
    )
end

function run_aerospace_disturbance_batch(;
    scenarios=[:baseline, :propeller_failure, :variable_thrust, :wind_gust, :aero_damage, :combined_failure],
    Ntraj=100,
    t_final=10.0,
    max_GPUs=0,
    quiet=true,
    kwargs...
)
    results = Dict{Symbol, Any}()

    for scenario in scenarios
        @info "Running aerospace disturbance scenario" scenario=scenario

        if scenario == :baseline
            results[scenario] = run_uncertain_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
                wind_profile=NO_WIND_PROFILE,
                fault_profile=DEFAULT_FAULT_PROFILE,
                aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
                plant_mismatch_enabled=false,
                output_tag="baseline",
            )
        elseif scenario == :propeller_failure
            results[scenario] = run_propeller_failure_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
            )
        elseif scenario == :variable_thrust
            results[scenario] = run_variable_thrust_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
            )
        elseif scenario == :wind_gust
            results[scenario] = run_wind_gust_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
            )
        elseif scenario == :aero_damage
            results[scenario] = run_aero_damage_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
            )
        elseif scenario == :combined_failure
            results[scenario] = run_combined_failure_ensemble_and_plot(;
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                quiet=quiet,
                kwargs...,
            )
        else
            error("Unknown scenario: $scenario")
        end
    end

    return results
end

function _run_controller_case(; controller_type, systems, seed=nothing, kwargs...)
    seed !== nothing && Random.seed!(seed)

    setup = nothing
    solutions = nothing
    timing = @timed begin
        setup, solutions = main(;
            kwargs...,
            systems=systems,
            controller_type=controller_type,
        )
    end

    return setup, solutions, timing
end

function _controller_metric_row(label, timing, Ntraj)
    return (
        controller = label,
        wall_seconds = timing.time,
        gc_seconds = timing.gctime,
        allocated_mb = timing.bytes / 1024^2,
        seconds_per_trajectory = timing.time / Ntraj,
    )
end

function run_lqr_ilqr_l1_comparison_and_plot(;
    Ntraj=100,
    t_final=10.0,
    max_GPUs=0,
    dt=1e-2,
    save_stride=5,
    ilqr_horizon=80,
    ilqr_max_iter=5,
    brownian_sigma=0.15,
    aleatoric_sigma=0.15,
    L1_omega=10.0,
    L1_sample_steps=25,
    L1_predictor_lambda=25.0,
    L1_control_reg=1e-3,
    L1_control_limit=2.0,
    initial_mean_offsets=DEFAULT_INITIAL_MEAN_OFFSETS,
    wind_profile=NO_WIND_PROFILE,
    fault_profile=PROPELLER_FAILURE_PROFILE,
    aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
    plant_mismatch_enabled=true,
    output_tag="lqr_ilqr_l1_propeller_failure",
    plot_title="LQR vs iLQR vs LQR + L1-DRAC: Propeller Failure",
    seed=598,
    quiet=true,
)
    common_kwargs = (
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        L1_omega=L1_omega,
        L1_sample_steps=L1_sample_steps,
        L1_predictor_lambda=L1_predictor_lambda,
        L1_control_reg=L1_control_reg,
        L1_control_limit=L1_control_limit,
        initial_mean_offsets=initial_mean_offsets,
        wind_profile=wind_profile,
        fault_profile=fault_profile,
        aero_damage_profile=aero_damage_profile,
        plant_mismatch_enabled=plant_mismatch_enabled,
        quiet=quiet,
    )

    @info "Running controller comparison case" controller="LQR" output_tag=output_tag
    setup_lqr, sol_lqr, timing_lqr = _run_controller_case(;
        common_kwargs...,
        controller_type=:lqr,
        systems=[:true_sys],
        seed=seed,
    )

    @info "Running controller comparison case" controller="iLQR" output_tag=output_tag
    setup_ilqr, sol_ilqr, timing_ilqr = _run_controller_case(;
        common_kwargs...,
        controller_type=:ilqr,
        systems=[:true_sys],
        seed=seed,
    )

    @info "Running controller comparison case" controller="LQR + L1-DRAC" output_tag=output_tag
    setup_lqr_l1, sol_lqr_l1, timing_lqr_l1 = _run_controller_case(;
        common_kwargs...,
        controller_type=:lqr,
        systems=[:L1_sys],
        seed=seed,
    )

    base_path = joinpath(@__DIR__, "quad_ilqr_sol_logs", string(output_tag))
    lqr_path = joinpath(base_path, "lqr")
    ilqr_path = joinpath(base_path, "ilqr")
    lqr_l1_path = joinpath(base_path, "lqr_l1")

    log_state_results(setup_lqr, sol_lqr; path=lqr_path)
    log_state_results(setup_ilqr, sol_ilqr; path=ilqr_path)
    log_state_results(setup_lqr_l1, sol_lqr_l1; path=lqr_l1_path)

    metrics = (
        lqr = _controller_metric_row("LQR", timing_lqr, Ntraj),
        ilqr = _controller_metric_row("iLQR", timing_ilqr, Ntraj),
        lqr_l1 = _controller_metric_row("LQR + L1-DRAC", timing_lqr_l1, Ntraj),
    )
    @info "Controller comparison metrics" metrics=metrics

    figs = generate_controller_comparison_plots(;
        lqr_path=lqr_path,
        ilqr_path=ilqr_path,
        lqr_l1_path=lqr_l1_path,
        output_prefix="quad_ilqr_$(output_tag)",
        plot_title=plot_title,
    )
    resource_fig = generate_controller_resource_plot(;
        metrics=metrics,
        output_prefix="quad_ilqr_$(output_tag)",
        plot_title="$(plot_title) computational resources",
    )

    return (figs=figs, resource_fig=resource_fig, metrics=metrics)
end

function run_lqr_ilqr_l1_baseline_comparison(; output_tag="lqr_ilqr_l1_baseline", kwargs...)
    return run_lqr_ilqr_l1_comparison_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=DEFAULT_FAULT_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        plant_mismatch_enabled=false,
        output_tag=output_tag,
        plot_title="LQR vs iLQR vs LQR + L1-DRAC: $(scenario_title(:baseline))",
    )
end

function run_lqr_ilqr_l1_propeller_failure_comparison(; output_tag="lqr_ilqr_l1_propeller_failure", kwargs...)
    return run_lqr_ilqr_l1_comparison_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=PROPELLER_FAILURE_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        output_tag=output_tag,
        plot_title="LQR vs iLQR vs LQR + L1-DRAC: $(scenario_title(:propeller_failure))",
    )
end

function run_lqr_ilqr_l1_aero_damage_comparison(; output_tag="lqr_ilqr_l1_aero_damage", kwargs...)
    return run_lqr_ilqr_l1_comparison_and_plot(;
        kwargs...,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=DEFAULT_FAULT_PROFILE,
        aero_damage_profile=DAMAGED_AIRFRAME_PROFILE,
        output_tag=output_tag,
        plot_title="LQR vs iLQR vs LQR + L1-DRAC: $(scenario_title(:aero_damage))",
    )
end

function run_lqr_ilqr_l1_baseline_and_propeller_batch(;
    Ntraj=100,
    t_final=10.0,
    max_GPUs=0,
    quiet=true,
    kwargs...
)
    results = Dict{Symbol, Any}()

    @info "Running LQR/iLQR/LQR+L1 controller comparison" scenario=:baseline
    results[:baseline] = run_lqr_ilqr_l1_baseline_comparison(;
        Ntraj=Ntraj,
        t_final=t_final,
        max_GPUs=max_GPUs,
        quiet=quiet,
        kwargs...,
    )

    @info "Running LQR/iLQR/LQR+L1 controller comparison" scenario=:propeller_failure
    results[:propeller_failure] = run_lqr_ilqr_l1_propeller_failure_comparison(;
        Ntraj=Ntraj,
        t_final=t_final,
        max_GPUs=max_GPUs,
        quiet=quiet,
        kwargs...,
    )

    return results
end

function _is_single_solution(sol_vec)
    return sol_vec !== nothing && !isempty(sol_vec) && hasproperty(sol_vec[1], :t)
end

function _single_solution_data(sol; state_dim=nothing)
    t = sol.t
    u = state_dim === nothing ? sol.u : [state[1:state_dim] for state in sol.u]
    mean = u
    var = [zero(state) for state in u]
    return (t=t, u=[u], mean=mean, var=var)
end

function _save_solution_data(path, filename, data)
    file_path = joinpath(path, filename)
    jldsave(file_path; t=data.t, u=data.u, mean=data.mean, var=data.var)
    return file_path
end

function log_state_results(setup, solutions; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"))
    mkpath(path)

    if _is_single_solution(solutions.nominal_sol) ||
       _is_single_solution(solutions.true_sol) ||
       _is_single_solution(solutions.L1_sol)

        nominal_path = nothing
        true_path = nothing
        L1_path = nothing

        if _is_single_solution(solutions.nominal_sol)
            data = _single_solution_data(solutions.nominal_sol[1])
            nominal_path = _save_solution_data(path, "states_nominal.jld2", data)
        end

        if _is_single_solution(solutions.true_sol)
            data = _single_solution_data(solutions.true_sol[1])
            true_path = _save_solution_data(path, "states_true.jld2", data)
        end

        if _is_single_solution(solutions.L1_sol)
            data = _single_solution_data(solutions.L1_sol[1]; state_dim=setup.system_dimensions.n)
            L1_path = _save_solution_data(path, "states_L1.jld2", data)
        end

        return (nominal=nominal_path, true_sys=true_path, L1=L1_path)
    end

    return state_logging(setup.system_dimensions;
        sol_nominal=solutions.nominal_sol,
        sol_true=solutions.true_sol,
        sol_L1=solutions.L1_sol,
        path=path)
end

include("plotting_utils.jl")

function generate_position_plot(; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"), max_traj=1000,
                                filename="quad_ilqr_position_cloud.png")
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    position_fig = plot_position_results(nom, tru, L1; max_traj=max_traj)
    savefig(position_fig, joinpath(@__DIR__, filename))
    @info "Saved $filename"
    return position_fig
end

function generate_nominal_position_plots(; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"),
                                         max_traj=1000,
                                         time_filename="quad_ilqr_nominal_position_plot.png",
                                         xyz_filename="quad_ilqr_nominal_xyz_paths.png")
    nom = load(joinpath(path, "states_nominal.jld2"))

    setup = setup_system(; Ntraj=1)
    x_goal = setup.nominal_system.nom_vec_fields.dynamics_params.x_goal

    position_fig = plot_nominal_position_results(nom; x_goal=x_goal, max_traj=max_traj)
    savefig(position_fig, joinpath(@__DIR__, time_filename))
    @info "Saved $time_filename"

    xyz_fig = plot_nominal_xyz_paths(nom; x_goal=x_goal, max_traj=max_traj)
    savefig(xyz_fig, joinpath(@__DIR__, xyz_filename))
    @info "Saved $xyz_filename"

    return position_fig, xyz_fig
end

function generate_adaptive_comparison_plots(; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"),
                                            max_traj=1000,
                                            output_prefix="quad_ilqr_nominal_vs_L1",
                                            plot_title="Nominal vs L1-DRAC")
    nom = load(joinpath(path, "states_nominal.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_baseline_adaptive_results(nom, L1; max_traj=max_traj,
                                         plot_title="$(plot_title) trajectory clouds")
    states_filename = "$(output_prefix)_states_plot.png"
    savefig(fig, joinpath(@__DIR__, states_filename))
    @info "Saved $states_filename"

    position_fig = plot_baseline_adaptive_position_results(nom, L1; max_traj=max_traj,
                                                           plot_title="$(plot_title) position trajectory clouds")
    position_filename = "$(output_prefix)_position_plot.png"
    savefig(position_fig, joinpath(@__DIR__, position_filename))
    @info "Saved $position_filename"

    return fig, position_fig
end

function generate_three_system_comparison_plots(; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"),
                                                max_traj=1000,
                                                output_prefix="quad_ilqr_three_system",
                                                plot_title="Nominal vs True Disturbed vs L1-DRAC",
                                                output_dir=@__DIR__)
    mkpath(output_dir)
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_results(nom, tru, L1; max_traj=max_traj,
                       plot_title="$(plot_title) trajectory clouds")
    states_filename = "$(output_prefix)_states_plot.png"
    savefig(fig, joinpath(output_dir, states_filename))
    @info "Saved $states_filename"

    position_fig = plot_position_results(nom, tru, L1; max_traj=max_traj,
                                         plot_title="$(plot_title) position trajectory clouds")
    position_filename = "$(output_prefix)_position_plot.png"
    savefig(position_fig, joinpath(output_dir, position_filename))
    @info "Saved $position_filename"

    return fig, position_fig
end

function generate_controller_comparison_plots(; lqr_path,
                                              ilqr_path,
                                              lqr_l1_path,
                                              max_traj=1000,
                                              output_prefix="quad_ilqr_lqr_ilqr_l1",
                                              plot_title="LQR vs iLQR vs LQR + L1-DRAC")
    lqr = load(joinpath(lqr_path, "states_true.jld2"))
    ilqr = load(joinpath(ilqr_path, "states_true.jld2"))
    lqr_l1 = load(joinpath(lqr_l1_path, "states_L1.jld2"))

    fig = plot_controller_comparison_results(lqr, ilqr, lqr_l1; max_traj=max_traj,
                                             plot_title="$(plot_title) trajectory clouds")
    states_filename = "$(output_prefix)_states_plot.png"
    savefig(fig, joinpath(@__DIR__, states_filename))
    @info "Saved $states_filename"

    position_fig = plot_controller_comparison_position_results(lqr, ilqr, lqr_l1;
                                                               max_traj=max_traj,
                                                               plot_title="$(plot_title) position trajectory clouds")
    position_filename = "$(output_prefix)_position_plot.png"
    savefig(position_fig, joinpath(@__DIR__, position_filename))
    @info "Saved $position_filename"

    return fig, position_fig
end

function generate_controller_resource_plot(; metrics,
                                           output_prefix="quad_ilqr_lqr_ilqr_l1",
                                           plot_title="Controller computational resources")
    resource_fig = plot_controller_resource_metrics(metrics; plot_title=plot_title)
    resource_filename = "$(output_prefix)_resource_plot.png"
    savefig(resource_fig, joinpath(@__DIR__, resource_filename))
    @info "Saved $resource_filename"

    return resource_fig
end

function generate_state_plots(; path=joinpath(@__DIR__, "quad_ilqr_sol_logs"), max_traj=1000)
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_results(nom, tru, L1; max_traj=max_traj)
    savefig(fig, joinpath(@__DIR__, "quad_ilqr_states_plot.png"))
    @info "Saved quad_ilqr_states_plot.png"

    position_fig = generate_position_plot(; path=path, max_traj=max_traj,
                                          filename="quad_ilqr_position_plot.png")

    return fig, position_fig
end
