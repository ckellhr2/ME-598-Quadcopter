## In the package as examples/ex1/doubleintegrator1D.jl 

using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using StaticArrays
using Plots
using JLD2

###################################################################
## SYSTEM SETUP
###################################################################
function setup_system(; Ntraj=10) # Ntraj = number of trajectories for ensemble sims, default val 10
    # Simulation Parameters
    tspan = (0.0, 5.0)
    Δₜ = 1e-4
    Δ_saveat = 1e2 * Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions: n=12 states, m=4 inputs (motor thrusts), d=1 noise channel (deterministic)
    n, m, d = 12, 4, 1
    system_dimensions = sys_dims(n, m, d)

    # State layout: [px,py,pz, vx,vy,vz, φ,θ,ψ, p_r,q_r,r_r]
    #   position (1-3), velocity (4-6), Euler angles (7-9), body rates (10-12)
    # Input layout: [T1, T2, T3, T4]  (motor thrusts, N)
    #   + config: T1=front, T2=right, T3=back, T4=left

    # Physical parameters
    m_q   = 0.5      # mass (kg)
    g_acc = 9.81     # gravity (m/s²)
    Ix    = 4.0e-3   # roll inertia (kg⋅m²)
    Iy    = 4.0e-3   # pitch inertia (kg⋅m²)
    Iz    = 8.0e-3   # yaw inertia (kg⋅m²)
    l_arm = 0.17     # motor arm length (m)
    kτ    = 0.016    # reaction torque / thrust ratio

    # Cascaded PD hover controller gains
    # Outer loop (position → desired attitude)
    Kx = 2.0;  Kdx = 2.0
    Ky = 2.0;  Kdy = 2.0
    Kz = 5.0;  Kdz = 3.0
    # Inner loop (attitude/rate → torques)
    Kφ  = 80.0;  Kωp = 10.0
    Kθ  = 80.0;  Kωq = 10.0
    Kψ  = 40.0;  Kωr = 5.0

    # dp contains only scalar Float64 values (no SMatrix) for GPU isbits compatibility
    dp = (; m_q, g_acc, Ix, Iy, Iz, l_arm, kτ,
            Kx, Kdx, Ky, Kdy, Kz, Kdz,
            Kφ, Kωp, Kθ, Kωq, Kψ, Kωr)

    # Cascaded PD: position error → thrust commands
    # T = eltype(x) ensures all arithmetic matches the state vector type (Float32 on GPU)
    # Mixing matrix inverse applied analytically (+ config: T1=front,T2=right,T3=back,T4=left):
    #   T1 = F/4 - τy/(2l) + τz/(4kτ)
    #   T2 = F/4 + τx/(2l) - τz/(4kτ)
    #   T3 = F/4 + τy/(2l) + τz/(4kτ)
    #   T4 = F/4 - τx/(2l) - τz/(4kτ)
    function baseline_input(t, x, dp)
        T = eltype(x)
        px, py, pz      = x[1], x[2], x[3]
        vx, vy, vz      = x[4], x[5], x[6]
        φ,  θ,  ψ       = x[7], x[8], x[9]
        p_r, q_r, r_r   = x[10], x[11], x[12]

        m_q  = T(dp.m_q);  g_acc = T(dp.g_acc)
        Kx   = T(dp.Kx);   Kdx   = T(dp.Kdx)
        Ky   = T(dp.Ky);   Kdy   = T(dp.Kdy)
        Kz   = T(dp.Kz);   Kdz   = T(dp.Kdz)
        Kφ   = T(dp.Kφ);   Kωp   = T(dp.Kωp)
        Kθ   = T(dp.Kθ);   Kωq   = T(dp.Kωq)
        Kψ   = T(dp.Kψ);   Kωr   = T(dp.Kωr)
        l    = T(dp.l_arm); kτ   = T(dp.kτ)

        # Outer loop: desired total thrust and attitude (hover at origin)
        F_des = m_q * (g_acc - Kz*pz - Kdz*vz)
        φ_des = (Ky*py + Kdy*vy) / g_acc
        θ_des = -(Kx*px + Kdx*vx) / g_acc

        # Inner loop: attitude error → body torques
        τ_x = Kφ*(φ_des - φ) - Kωp*p_r
        τ_y = Kθ*(θ_des - θ) - Kωq*q_r
        τ_z = -Kψ*ψ - Kωr*r_r

        # Analytical inverse of mixing matrix
        T1 = F_des/4 - τ_y/(2*l) + τ_z/(4*kτ)
        T2 = F_des/4 + τ_x/(2*l) - τ_z/(4*kτ)
        T3 = F_des/4 + τ_y/(2*l) + τ_z/(4*kτ)
        T4 = F_des/4 - τ_x/(2*l) - τ_z/(4*kτ)
        return SVector{4,T}(T1, T2, T3, T4)
    end

    # Nominal Vector Fields

    # f: full nonlinear drift with baseline controller baked in
    function f(t, x, dp)
        T = eltype(x)
        vx, vy, vz      = x[4], x[5], x[6]
        φ,  θ,  ψ       = x[7], x[8], x[9]
        p_r, q_r, r_r   = x[10], x[11], x[12]

        cφ, sφ = cos(φ), sin(φ)
        cθ, sθ = cos(θ), sin(θ)
        cψ, sψ = cos(ψ), sin(ψ)
        tθ     = tan(θ)

        m_q   = T(dp.m_q);  g_acc = T(dp.g_acc)
        l_arm = T(dp.l_arm); kτ   = T(dp.kτ)
        Ix    = T(dp.Ix);   Iy    = T(dp.Iy);  Iz = T(dp.Iz)

        # Inertial-frame thrust direction (body z rotated to inertial)
        fx_rot = cψ*sθ*cφ + sψ*sφ
        fy_rot = sψ*sθ*cφ - cψ*sφ
        fz_rot = cθ*cφ

        u = baseline_input(t, x, dp)
        F_total = u[1] + u[2] + u[3] + u[4]
        τ_x = l_arm*(u[2] - u[4])
        τ_y = l_arm*(u[3] - u[1])
        τ_z = kτ*(u[1] - u[2] + u[3] - u[4])

        dpx = vx;  dpy = vy;  dpz = vz
        dvx = fx_rot * F_total / m_q
        dvy = fy_rot * F_total / m_q
        dvz = fz_rot * F_total / m_q - g_acc

        dφ  = p_r + (q_r*sφ + r_r*cφ)*tθ
        dθ  = q_r*cφ - r_r*sφ
        dψ  = (q_r*sφ + r_r*cφ) / cθ

        dp_r = (Iy - Iz)/Ix * q_r*r_r + τ_x/Ix
        dq_r = (Iz - Ix)/Iy * p_r*r_r + τ_y/Iy
        dr_r = (Ix - Iy)/Iz * p_r*q_r + τ_z/Iz

        return SVector{12,T}(dpx, dpy, dpz, dvx, dvy, dvz, dφ, dθ, dψ, dp_r, dq_r, dr_r)
    end

    # g: state-dependent input matrix B(x), shape 12×4 (SMatrix{12,4})
    # Maps additional thrust commands [δT1,δT2,δT3,δT4] → ẋ for L1 adaptive signal
    function g(t, x, dp)
        T = eltype(x)
        φ, θ, ψ = x[7], x[8], x[9]
        cφ, sφ = cos(φ), sin(φ)
        cθ, sθ = cos(θ), sin(θ)
        cψ, sψ = cos(ψ), sin(ψ)

        fx_row = (cψ*sθ*cφ + sψ*sφ) / T(dp.m_q)
        fy_row = (sψ*sθ*cφ - cψ*sφ) / T(dp.m_q)
        fz_row = cθ*cφ / T(dp.m_q)
        l_Ix   = T(dp.l_arm) / T(dp.Ix)
        l_Iy   = T(dp.l_arm) / T(dp.Iy)
        kτ_Iz  = T(dp.kτ)    / T(dp.Iz)
        z      = zero(T)

        # Column-major layout: each block of 12 = one column (T1, T2, T3, T4)
        return SMatrix{12,4,T}(
            # T1: τx=0, τy=-l, τz=+kτ
            z, z, z, fx_row, fy_row, fz_row, z, z, z,  z,     -l_Iy,  kτ_Iz,
            # T2: τx=+l, τy=0, τz=-kτ
            z, z, z, fx_row, fy_row, fz_row, z, z, z,  l_Ix,  z,     -kτ_Iz,
            # T3: τx=0, τy=+l, τz=+kτ
            z, z, z, fx_row, fy_row, fz_row, z, z, z,  z,     l_Iy,   kτ_Iz,
            # T4: τx=-l, τy=0, τz=-kτ
            z, z, z, fx_row, fy_row, fz_row, z, z, z, -l_Ix,  z,     -kτ_Iz
        )
    end

    # g_perp: constant 12×8 matrix spanning directions unactuated at hover.
    # At hover (φ=θ=ψ=0), g's column space is within span{e6, e10, e11, e12},
    # so the complement is span{e1,e2,e3,e4,e5,e7,e8,e9} (identity cols 1-5, 7-9).
    # Column-major: each 12-element block is one column of the 12×8 matrix.
    function g_perp(_, x, _)
        T = eltype(x)
        z = zero(T);  o = one(T)
        return SMatrix{12,8,T}(
            o, z, z, z, z, z, z, z, z, z, z, z,  # e1
            z, o, z, z, z, z, z, z, z, z, z, z,  # e2
            z, z, o, z, z, z, z, z, z, z, z, z,  # e3
            z, z, z, o, z, z, z, z, z, z, z, z,  # e4
            z, z, z, z, o, z, z, z, z, z, z, z,  # e5
            z, z, z, z, z, z, o, z, z, z, z, z,  # e7
            z, z, z, z, z, z, z, o, z, z, z, z,  # e8
            z, z, z, z, z, z, z, z, o, z, z, z   # e9
        )
    end

    # p: nominal stochastic diffusion (zero for deterministic model)
    p(t, x, dp) = zero(SMatrix{12,1,eltype(x),12})

    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    # Uncertain Vector Fields (placeholder — small physics-motivated perturbations)
    function Λμ(_, x, _)
        T = eltype(x)
        return T(1e-2) * SVector{12,T}(
            zero(T), zero(T), zero(T),
            sin(x[4]), sin(x[5]), one(T) + cos(x[6]),
            sin(x[7]), sin(x[8]), zero(T),
            one(T) + x[10]^2, one(T) + x[11]^2, one(T) + x[12]^2
        )
    end

    Λσ(t, x, dp) = zero(SMatrix{12,1,eltype(x),12})

    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial Distributions (near hover at z=2 m)
    x0_nominal = [0.0, 0.0, 2.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0]
    x0_true    = [0.3, 0.3, 2.5,  0.1, 0.1, 0.0,  0.05, 0.05, 0.0, 0.0, 0.0, 0.0]
    nominal_ξ₀ = MvNormal(x0_nominal, 1e-4 * I(12))
    true_ξ₀    = MvNormal(x0_true,    1e-3 * I(12))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define Systems
    nominal_system = nom_sys(system_dimensions, nominal_components,
                        initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components,
                        uncertain_components, initial_distributions)

    # L1-DRAC Parameters (PLACEHOLDER values)
    ω  = 50.0       # filter bandwidth
    Tₛ = 10 * Δₜ   # sample time (integer multiple of Δₜ)
    λₛ = 100.0      # predictor stability
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system = nominal_system,
        true_system = true_system,
        L1params = L1params,
        system_dimensions = system_dimensions
    )
end

###################################################################
## MAIN
###################################################################
function main(; Ntraj = Int(1e1), max_GPUs=10, systems=[:nominal_sys, :true_sys, :L1_sys]) 

    @info "Warmup run for JIT compilation"
    println("=====================================") 
    warmup_setup = setup_system(; Ntraj = 10)
    run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems);

    println("=====================================")
    @info "Complete run for Ntraj=$Ntraj" 
    println("=====================================")
    setup = setup_system(; Ntraj = Ntraj)
    solutions = run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return setup, solutions
end


## SOLVING
setup, solutions = main();
nominal_sol, true_sol, L1_sol = solutions;

###################################################################
## DATA LOGGING
###################################################################
# Wrapper
function log_state_results(setup, solutions; path=joinpath(@__DIR__, "sol_logs"))
    state_logging(setup.system_dimensions;
        sol_nominal=solutions.nominal_sol,
        sol_true=solutions.true_sol,
        sol_L1=solutions.L1_sol,
        path=path)
end

# Save simulation data to JLD2 files
log_state_results(setup, solutions)

###################################################################
## PLOTS
###################################################################
include("plotting_utils.jl")

function generate_state_plots(; path=joinpath(@__DIR__, "sol_logs"), max_traj=500)
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_results(nom, tru, L1; max_traj=max_traj)
    savefig(fig, joinpath(@__DIR__, "states_plot.png"))
    @info "Saved states_plot.png"
    return fig
end

# Generate and save state trajectory plots
generate_state_plots(; max_traj=500)