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

const NX_QUAD = 12
const NU_QUAD = 4
const GRAVITY = 9.81
const ARM_LENGTH = 0.15
const K_YAW = 0.01
const ILQR_REG = 1e-6

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
    ix, iy, iz = inertia_diag
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

function cost_stage(x, delta_u, x_goal, Q, R)
    dx = x - x_goal
    return dot(dx, Q * dx) + dot(delta_u, R * delta_u)
end

function cost_final(x, x_goal, Qf)
    dx = x - x_goal
    return dot(dx, Qf * dx)
end

function rollout_dynamics(x0, delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover)
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
        total_cost += cost_stage(x_seq[k, :], delta_u_clipped, x_goal, Q, R)
    end

    total_cost += cost_final(x_seq[end, :], x_goal, Qf)
    return x_seq, u_abs_seq, total_cost
end

function ilqr(x0, x_goal, horizon, dt, mass, inertia_diag, Q, R, Qf;
              u_init=nothing, max_iter=5)
    hover = mass * GRAVITY / 4.0
    if u_init === nothing
        delta_u_seq = zeros(horizon, NU_QUAD)
    else
        delta_u_seq = clamp.(u_init .- hover, -hover, 1.5 * hover)
    end

    for _ in 1:max_iter
        x_seq, _, current_cost = rollout_dynamics(
            x0, delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover
        )

        Vx = 2.0 .* (Qf * (x_seq[end, :] - x_goal))
        Vxx = 2.0 .* Qf
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
                x0, trial_delta_u_seq, x_goal, dt, mass, inertia_diag, Q, R, Qf, hover
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

function quadrotor_ilqr_input(t, x, dp)
    u_seq = ilqr(
        collect(x),
        dp.x_goal,
        dp.horizon,
        dp.ilqr_dt,
        dp.mass_nominal,
        dp.inertia_nominal,
        dp.Q,
        dp.R,
        dp.Qf;
        u_init=dp.u_hover_init,
        max_iter=dp.ilqr_max_iter,
    )
    return u_seq[1, :]
end

function quadrotor_nominal_drift(t, x, dp)
    yaw = x[9]
    _, Bdt = build_linearized_dynamics(1.0, dp.mass_nominal, dp.inertia_nominal, yaw)
    u = quadrotor_ilqr_input(t, x, dp)
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

    u = quadrotor_ilqr_input(t, x, dp)
    delta_u = u .- dp.hover_nominal
    return (B_true - B_nominal) * delta_u
end

function setup_system(; Ntraj=2, t_final=1.0, dt=1e-2, save_stride=5,
                      ilqr_horizon=120, ilqr_max_iter=5,
                      brownian_sigma=0.12, aleatoric_sigma=0.18)
    tspan = (0.0, t_final)
    saveat = save_stride * dt
    simulation_parameters = sim_params(tspan, dt, Ntraj, saveat)

    n, m, d = NX_QUAD, NU_QUAD, NU_QUAD
    system_dimensions = sys_dims(n, m, d)

    mass_nominal = 1.0
    inertia_nominal = [0.005, 0.005, 0.009]
    plant_mass_scale = 1.15
    plant_inertia_scale = 1.10
    plant_thrust_scale = 0.90
    plant_yaw_coeff_scale = 1.20

    Q, R, Qf = make_cost_matrices()
    x_goal = zeros(NX_QUAD)
    x_goal[1:3] .= [1.3, 2.1, 1.4]
    hover_nominal = mass_nominal * GRAVITY / 4.0

    dp = (
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
        hover_nominal = hover_nominal,
        u_hover_init = fill(1.01 * hover_nominal, ilqr_horizon, NU_QUAD),
        horizon = ilqr_horizon,
        ilqr_dt = 1 / 240,
        ilqr_max_iter = ilqr_max_iter,
        B_nominal = continuous_input_matrix(mass_nominal, inertia_nominal),
    )

    f(t, x, dp) = quadrotor_nominal_drift(t, x, dp)
    g(t, x, dp) = dp.B_nominal
    g_perp(t, x, dp) = Matrix{Float64}(I, n, n)[:, [1, 2, 3, 4, 5, 7, 8, 9]]

    brownian_diffusion = zeros(n, d)
    brownian_diffusion[4, 1] = brownian_sigma
    brownian_diffusion[5, 2] = brownian_sigma
    brownian_diffusion[6, 3] = 0.5 * brownian_sigma
    brownian_diffusion[12, 4] = 0.25 * brownian_sigma

    p(t, x, dp) = brownian_diffusion
    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    Lambda_mu(t, x, dp) = zeros(n)
    aleatoric_diffusion = zeros(n, d)
    aleatoric_diffusion[4, 1] = aleatoric_sigma
    aleatoric_diffusion[5, 2] = aleatoric_sigma
    aleatoric_diffusion[6, 3] = 0.5 * aleatoric_sigma
    aleatoric_diffusion[10, 1] = 0.2 * aleatoric_sigma
    aleatoric_diffusion[11, 2] = 0.2 * aleatoric_sigma
    aleatoric_diffusion[12, 4] = 0.25 * aleatoric_sigma

    Lambda_sigma(t, x, dp) = aleatoric_diffusion
    uncertain_components = uncertain_vector_fields(Lambda_mu, Lambda_sigma)

    nominal_mu = zeros(n)
    nominal_mu[1:3] .= [0.0, 0.0, 1.0]
    true_mu = copy(nominal_mu)
    initial_cov = Diagonal(vcat(fill(0.05, 3), fill(0.02, 3), fill(0.01, 6)))
    initial_distributions = init_dist(
        MvNormal(nominal_mu, initial_cov),
        MvNormal(true_mu, initial_cov),
    )

    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, uncertain_components, initial_distributions)

    omega = 50.0
    sample_time = 10 * dt
    predictor_lambda = 100.0
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

function main(; Ntraj=2, max_GPUs=0, systems=[:nominal_sys, :true_sys, :L1_sys],
              t_final=1.0, dt=1e-2, save_stride=5,
              ilqr_horizon=120, ilqr_max_iter=5,
              brownian_sigma=0.12, aleatoric_sigma=0.18,
              quiet=true)
    quiet || @info "Warmup run for JIT compilation"
    warmup_setup = setup_system(;
        Ntraj=1,
        t_final=min(t_final, 0.1),
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
    )
    _run_simulations_maybe_quiet(warmup_setup; max_GPUs=max_GPUs, systems=systems, quiet=quiet);

    quiet || @info "Complete run" Ntraj=Ntraj t_final=t_final dt=dt ilqr_horizon=ilqr_horizon ilqr_max_iter=ilqr_max_iter
    setup = setup_system(;
        Ntraj=Ntraj,
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
    )
    solutions = _run_simulations_maybe_quiet(setup; max_GPUs=max_GPUs, systems=systems, quiet=quiet)
    return setup, solutions
end

function run_long_ensemble(; Ntraj=1000, t_final=5.0, max_GPUs=0,
                           dt=2e-2, save_stride=5,
                           ilqr_horizon=120, ilqr_max_iter=5,
                           brownian_sigma=0.12, aleatoric_sigma=0.18,
                           quiet=true)
    setup, solutions = main(;
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        quiet=quiet,
    )
    log_state_results(setup, solutions)
    return generate_state_plots()
end

function run_nominal_ensemble_and_plot(; Ntraj=50, t_final=10.0, max_GPUs=0,
                                       dt=1e-2, save_stride=5,
                                       ilqr_horizon=120, ilqr_max_iter=5,
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
        quiet=quiet,
    )
    log_state_results(setup, solutions)
    return generate_nominal_position_plots()
end

function run_uncertain_ensemble_and_plot(; Ntraj=50, t_final=10.0, max_GPUs=0,
                                        dt=1e-2, save_stride=5,
                                        ilqr_horizon=120, ilqr_max_iter=5,
                                        brownian_sigma=0.12,
                                        aleatoric_sigma=0.18,
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
        brownian_sigma=brownian_sigma,
        aleatoric_sigma=aleatoric_sigma,
        quiet=quiet,
    )
    log_state_results(setup, solutions)
    return generate_state_plots()
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
