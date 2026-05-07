## deterministic_example.jl
## 12-state quadrotor simulation with iLQR trajectory-tracking controller.
## iLQR linearizes the discrete-time dynamics about the current trajectory
## iterate at each backward-pass step — NOT about a fixed equilibrium.

using L1DRAC
using LinearAlgebra
using Distributions
using StaticArrays
using Plots
using JLD2
using ForwardDiff

###################################################################
## QUADROTOR PHYSICAL PARAMETERS
###################################################################

# All scalars — GPU-safe inside NamedTuple dp
const PHYS = (
    m   = 0.5,      # kg    — vehicle mass
    g   = 9.81,     # m/s²  — gravitational acceleration
    Ixx = 4.9e-3,   # kg·m² — roll  moment of inertia
    Iyy = 4.9e-3,   # kg·m² — pitch moment of inertia
    Izz = 8.8e-3,   # kg·m² — yaw   moment of inertia
    l   = 0.225,    # m     — arm length (motor to CoM)
    km  = 0.01,     # —     — drag-torque to thrust ratio
)

const T_HOVER = PHYS.m * PHYS.g / 4   # per-motor hover thrust (N)

###################################################################
## NONLINEAR QUADROTOR DYNAMICS (continuous time)
###################################################################
#
# State  x ∈ ℝ¹²  [px  py  pz | vx  vy  vz | φ   θ   ψ | p   q   r]
#   (px,py,pz)  — position in inertial frame, Z-up
#   (vx,vy,vz)  — velocity in inertial frame
#   (φ, θ, ψ)   — ZYX Euler angles: roll, pitch, yaw
#   (p, q, r)   — body-frame angular rates
#
# Input  u ∈ ℝ⁴   [T1  T2  T3  T4]  motor thrusts in Newtons (≥ 0)
#   Motor layout (+ config, Z-up):
#     T1 — front (+x arm, CW  rotor → negative yaw reaction)
#     T2 — left  (+y arm, CCW rotor → positive yaw reaction)
#     T3 — back  (-x arm, CW  rotor → negative yaw reaction)
#     T4 — right (-y arm, CCW rotor → positive yaw reaction)
#
# Mixer:
#   F  = T1 + T2 + T3 + T4
#   τx = l·(T2 − T4)
#   τy = l·(T1 − T3)
#   τz = km·(−T1 + T2 − T3 + T4)

function quadrotor_ode(x::SVector{12,Tx}, u::SVector{4,Tu}, phys) where {Tx, Tu}
    # Promote to a single type T so the return is always SVector{12,T}.
    # This is critical on GPU: the state is Float32, phys constants are Float64.
    # Without this promotion, the mixed arithmetic produces a Float64 return while
    # the DiffEqGPU EM kernel expects a Float32 update, causing dynamic dispatch.
    T = promote_type(Tx, Tu)
    m, g_           = T(phys.m),   T(phys.g)
    Ixx, Iyy, Izz   = T(phys.Ixx), T(phys.Iyy), T(phys.Izz)
    l, km           = T(phys.l),   T(phys.km)

    px, py, pz, vx, vy, vz, φ, θ, ψ, bp, bq, br = SVector{12,T}(x)
    T1, T2, T3, T4 = SVector{4,T}(u)

    # Motor mixing → generalized forces
    F  =  T1 + T2 + T3 + T4
    τx =  l  * (T2 - T4)
    τy =  l  * (T1 - T3)
    τz =  km * (-T1 + T2 - T3 + T4)

    cφ, sφ = cos(φ), sin(φ)
    cθ, sθ = cos(θ), sin(θ)
    cψ, sψ = cos(ψ), sin(ψ)
    tθ     = tan(θ)

    # Inertial accelerations — thrust along body +Z rotated by ZYX rotation matrix
    # Column 3 of R_ZYX: [cψ·sθ·cφ+sψ·sφ, sψ·sθ·cφ−cψ·sφ, cθ·cφ]
    ax = (cψ*sθ*cφ + sψ*sφ) * (F/m)
    ay = (sψ*sθ*cφ - cψ*sφ) * (F/m)
    az =  cθ*cφ              * (F/m) - g_

    # Euler-angle kinematics (ZYX convention, body rates → Euler rates)
    φ̇ = bp + (bq*sφ + br*cφ) * tθ
    θ̇ =      bq*cφ - br*sφ
    ψ̇ =     (bq*sφ + br*cφ) / cθ

    # Euler equations (body-frame angular acceleration)
    ṗ = (Iyy - Izz)/Ixx * bq*br  +  τx/Ixx
    q̇ = (Izz - Ixx)/Iyy * bp*br  +  τy/Iyy
    ṙ = (Ixx - Iyy)/Izz * bp*bq  +  τz/Izz

    return SVector{12,T}(vx, vy, vz, ax, ay, az, φ̇, θ̇, ψ̇, ṗ, q̇, ṙ)
end

# Explicit unrolled 4×12 matrix-vector product.
# StaticArrays' _mul for matrices larger than ~8 columns falls back to a
# MVector-accumulator loop that GPUCompiler flags as a dynamic invocation.
# This helper uses only concrete scalar arithmetic and compiles cleanly on GPU.
@inline function _K_δx(K::SMatrix{4,12,T,48}, v::SVector{12,T}) where T
    SVector{4,T}(
        muladd(K[1,1],v[1], muladd(K[1,2],v[2], muladd(K[1,3],v[3], muladd(K[1,4],v[4],
        muladd(K[1,5],v[5], muladd(K[1,6],v[6], muladd(K[1,7],v[7], muladd(K[1,8],v[8],
        muladd(K[1,9],v[9], muladd(K[1,10],v[10], muladd(K[1,11],v[11], K[1,12]*v[12]))))))))))),
        muladd(K[2,1],v[1], muladd(K[2,2],v[2], muladd(K[2,3],v[3], muladd(K[2,4],v[4],
        muladd(K[2,5],v[5], muladd(K[2,6],v[6], muladd(K[2,7],v[7], muladd(K[2,8],v[8],
        muladd(K[2,9],v[9], muladd(K[2,10],v[10], muladd(K[2,11],v[11], K[2,12]*v[12]))))))))))),
        muladd(K[3,1],v[1], muladd(K[3,2],v[2], muladd(K[3,3],v[3], muladd(K[3,4],v[4],
        muladd(K[3,5],v[5], muladd(K[3,6],v[6], muladd(K[3,7],v[7], muladd(K[3,8],v[8],
        muladd(K[3,9],v[9], muladd(K[3,10],v[10], muladd(K[3,11],v[11], K[3,12]*v[12]))))))))))),
        muladd(K[4,1],v[1], muladd(K[4,2],v[2], muladd(K[4,3],v[3], muladd(K[4,4],v[4],
        muladd(K[4,5],v[5], muladd(K[4,6],v[6], muladd(K[4,7],v[7], muladd(K[4,8],v[8],
        muladd(K[4,9],v[9], muladd(K[4,10],v[10], muladd(K[4,11],v[11], K[4,12]*v[12])))))))))))
    )
end

###################################################################
## DISCRETE-TIME DYNAMICS — 4th-order Runge-Kutta
###################################################################

function rk4_step(x::SVector{12,Tx}, u::SVector{4,Tu}, dt, phys) where {Tx, Tu}
    k1 = quadrotor_ode(x,             u, phys)
    k2 = quadrotor_ode(x + (dt/2)*k1, u, phys)
    k3 = quadrotor_ode(x + (dt/2)*k2, u, phys)
    k4 = quadrotor_ode(x +  dt   *k3, u, phys)
    return x + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

###################################################################
## iLQR — discrete-time, finite-horizon
##
## Linearization in the backward pass is always performed about the
## CURRENT trajectory iterate (x_traj[k], u_traj[k]), never about a
## fixed equilibrium.  This is what distinguishes iLQR from standard LQR.
###################################################################

# Discrete Jacobians at (x̄, ū) via ForwardDiff — called only inside iLQR
function discrete_jacobians(x̄::SVector{12,Float64}, ū::SVector{4,Float64},
                              dt, phys)
    A = ForwardDiff.jacobian(x -> rk4_step(SVector{12}(x), ū, dt, phys), x̄)
    B = ForwardDiff.jacobian(u -> rk4_step(x̄, SVector{4}(u), dt, phys), ū)
    return SMatrix{12,12}(A), SMatrix{12,4}(B)
end

function ilqr_total_cost(x_traj, u_traj, x_ref, u_ref,
                          Q::SMatrix{12,12}, R::SMatrix{4,4}, Qf::SMatrix{12,12})
    J = 0.0
    for k in eachindex(u_traj)
        δx = x_traj[k] - x_ref[k]
        δu = u_traj[k] - u_ref[k]
        J += 0.5 * (dot(δx, Q*δx) + dot(δu, R*δu))
    end
    δxN = x_traj[end] - x_ref[end]
    return J + 0.5*dot(δxN, Qf*δxN)
end

# Backward pass: Riccati recursion over the current trajectory.
# Returns feedback gains K[k] (4×12) and feedforward terms d[k] (4-vec).
function ilqr_backward_pass(x_traj, u_traj, x_ref, u_ref,
                              Q::SMatrix{12,12}, R::SMatrix{4,4}, Qf::SMatrix{12,12},
                              dt, phys)
    N     = length(u_traj)
    K_all = Vector{SMatrix{4,12,Float64,48}}(undef, N)
    d_all = Vector{SVector{4,Float64}}(undef, N)

    # Terminal boundary condition
    δxN = x_traj[end] - x_ref[end]
    Vxx = Qf
    Vx  = Qf * δxN

    for k = N:-1:1
        # Linearize about the CURRENT trajectory point — core of iLQR
        Ak, Bk = discrete_jacobians(x_traj[k], u_traj[k], dt, phys)

        δxk = x_traj[k] - x_ref[k]
        δuk = u_traj[k] - u_ref[k]

        # Action-value (Q-function) expansions
        Qxx = Q  + Ak' * Vxx * Ak
        Quu = R  + Bk' * Vxx * Bk
        Qux =      Bk' * Vxx * Ak
        Qx  = Q  * δxk + Ak' * Vx
        Qu  = R  * δuk + Bk' * Vx

        Quu_reg = Quu + 1e-4 * I(4)   # Tikhonov regularization for PD guarantee

        K_all[k] = SMatrix{4,12}(-Quu_reg \ Qux)
        d_all[k] = SVector{4}(  -Quu_reg \ Qu  )

        # Value function update (efficient form: Vxx = Qxx + Qux'·K)
        Vxx = Qxx + Qux' * K_all[k]
        Vx  = Qx  + Qux' * d_all[k]
    end
    return K_all, d_all
end

# Forward pass with Armijo step size α
function ilqr_forward_pass(x0, x_traj, u_traj, K_all, d_all,
                             x_ref, u_ref, dt, phys; α=1.0,
                             u_min=0.0, u_max=6.0*T_HOVER)
    N     = length(u_traj)
    x_new = Vector{SVector{12,Float64}}(undef, N+1)
    u_new = Vector{SVector{4,Float64}}(undef, N)
    x_new[1] = x0
    for k = 1:N
        δx        = x_new[k] - x_traj[k]
        u_k       = u_traj[k] + K_all[k]*δx + α*d_all[k]
        u_new[k]  = SVector{4}(clamp.(u_k, u_min, u_max))
        x_new[k+1] = rk4_step(x_new[k], u_new[k], dt, phys)
    end
    return x_new, u_new
end

# Full iLQR solve — iterates backward/forward passes until convergence
function solve_ilqr(x0::SVector{12,Float64},
                     x_ref::Vector{SVector{12,Float64}},
                     u_ref::Vector{SVector{4,Float64}},
                     Q::SMatrix{12,12}, R::SMatrix{4,4}, Qf::SMatrix{12,12},
                     dt::Float64, phys;
                     max_iter=100, tol=1e-6)
    N = length(u_ref)
    @assert length(x_ref) == N + 1

    # Initial rollout using reference controls
    x_traj = Vector{SVector{12,Float64}}(undef, N+1)
    x_traj[1] = x0
    u_traj    = copy(u_ref)
    for k = 1:N
        x_traj[k+1] = rk4_step(x_traj[k], u_traj[k], dt, phys)
    end

    J_prev = ilqr_total_cost(x_traj, u_traj, x_ref, u_ref, Q, R, Qf)
    for _ = 1:max_iter
        K_all, d_all = ilqr_backward_pass(x_traj, u_traj, x_ref, u_ref,
                                           Q, R, Qf, dt, phys)
        # Armijo line search
        α       = 1.0
        accepted = false
        for _ = 1:10
            x_c, u_c = ilqr_forward_pass(x0, x_traj, u_traj, K_all, d_all,
                                          x_ref, u_ref, dt, phys; α=α)
            J_c = ilqr_total_cost(x_c, u_c, x_ref, u_ref, Q, R, Qf)
            if J_c ≤ J_prev
                x_traj, u_traj = x_c, u_c
                J_prev = J_c
                accepted = true
                break
            end
            α *= 0.5
        end
        !accepted && break
        abs(J_prev) < tol && break
    end

    K_final, _ = ilqr_backward_pass(x_traj, u_traj, x_ref, u_ref,
                                     Q, R, Qf, dt, phys)
    return x_traj, u_traj, K_final
end


###################################################################
## SYSTEM SETUP
###################################################################

function setup_system(; Ntraj=10)
    # Simulation parameters
    tspan    = (0.0, 5.0)
    Δₜ       = 1e-3          # simulation step (1 ms)
    Δ_saveat = 10 * Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System dimensions: n=12 states, m=4 inputs, d=2 disturbance modes (1 unmatched + 1 matched)
    n, m, d = 12, 4, 2
    system_dimensions = sys_dims(n, m, d)

    phys = PHYS

    # iLQR is solved offline at a coarser timestep for computational tractability
    # (zero-order-hold: the resulting gains are held constant within each iLQR interval)
    dt_ilqr = 0.02   # iLQR discretisation step (20 ms)
    N_ilqr  = Int((tspan[2] - tspan[1]) / dt_ilqr)   # 250 steps

    # Reference trajectory: hover at z = 1 m, all else zero
    x_ref_hover = SVector{12,Float64}(0,0,1, 0,0,0, 0,0,0, 0,0,0)
    u_ref_hover = SVector{4,Float64}(T_HOVER, T_HOVER, T_HOVER, T_HOVER)
    x_ref = fill(x_ref_hover, N_ilqr + 1)
    u_ref = fill(u_ref_hover, N_ilqr)

    # Quadratic cost: position/attitude errors penalised most
    Q  = SMatrix{12,12,Float64,144}(diagm(
            [20.0, 20.0, 20.0,    # position
              2.0,  2.0,  2.0,    # velocity
             10.0, 10.0,  5.0,    # angles
              0.5,  0.5,  0.5]))  # body rates
    R  = SMatrix{4,4,Float64,16}(0.05 * I(4))
    Qf = 10.0 * Q

    # Initial state: 0.3 m below hover altitude, small roll perturbation
    x0 = SVector{12,Float64}(0, 0, 0.7, 0, 0, 0, 0.05, 0, 0, 0, 0, 0)

    @info "Solving iLQR offline (N=$N_ilqr steps, dt=$dt_ilqr s)…"
    x_opt, u_opt, K_opt = solve_ilqr(x0, x_ref, u_ref, Q, R, Qf, dt_ilqr, phys)
    @info "iLQR solved."

    # DiffEqGPU batches every trajectory into a CuArray, so dp must be a bitstype:
    # all fields must be statically sized (SMatrix, SVector, plain scalars).
    # CuArray fields are heap-allocated and are NOT bitstypes — they fail this check.
    #
    # For a time-invariant hover reference, iLQR converges to a time-invariant gain.
    # K_opt[1] is the Riccati-converged gain from the backward pass — it was derived by
    # linearizing about the current trajectory iterate (iLQR), not about the equilibrium
    # directly.  Using it as a constant gain is therefore exact for hover.
    K_final = K_opt[1]                                          # SMatrix{4,12} — bitstype ✓
    u_eq    = SVector{4,Float64}(T_HOVER, T_HOVER, T_HOVER, T_HOVER)  # bitstype ✓
    x_eq    = x_ref_hover                                       # SVector{12}   — bitstype ✓

    dp = (; K_final, u_eq, x_eq, phys, u_max=6.0*T_HOVER)

    # iLQR feedback law — u = u_eq + K·(x − x_eq).
    # All dp fields are converted to T = eltype(x_raw) so the GPU kernel stays
    # in a single concrete Float32 world and avoids mixed-type dynamic dispatch.
    # _K_δx replaces K*δx to sidestep StaticArrays' large-matrix _mul path.
    function baseline_input(t, x_raw, dp)
        T   = eltype(x_raw)
        x_s = SVector{12,T}(x_raw)
        K   = SMatrix{4,12,T,48}(dp.K_final)
        u_e = SVector{4,T}(dp.u_eq)
        x_e = SVector{12,T}(dp.x_eq)
        u   = u_e + _K_δx(K, x_s - x_e)
        return SVector{4,T}(clamp.(u, zero(T), T(dp.u_max)))
    end

    # Closed-loop nominal dynamics — full nonlinear ODE driven by iLQR output
    f(t, x, dp) = quadrotor_ode(SVector{12}(x), baseline_input(t, x, dp), dp.phys)

    # Matched uncertainty direction: body thrust axis projected to velocity states (4–6).
    # All constants promoted to T to keep the return type SVector{12,T}.
    function g(t, x_raw, dp)
        T      = eltype(x_raw)
        φ, θ, ψ = T(x_raw[7]), T(x_raw[8]), T(x_raw[9])
        cφ, sφ  = cos(φ), sin(φ)
        cθ      = cos(θ)
        cψ, sψ  = cos(ψ), sin(ψ)
        m_inv   = T(inv(dp.phys.m))
        z       = zero(T)
        return SVector{12,T}(z, z, z,
                              (cψ*sθ*cφ + sψ*sφ)*m_inv,
                              (sψ*sθ*cφ - cψ*sφ)*m_inv,
                               cθ*cφ*m_inv,
                              z, z, z, z, z, z)
    end

    # Unmatched uncertainty direction — typed to match state element type
    function g_perp(t, x_raw, dp)
        T = eltype(x_raw)
        z = zero(T)
        return SVector{12,T}(z, z, z, one(T), z, z, z, z, z, z, z, z)
    end

    # p must be n×d (12×2) — the noise_rate_prototype in the GPU kernel is SMatrix{n,d}.
    # In the original 2-state example n=d=2 so d×n == n×d (square); for n=12, d=2 they differ.
    # p_um and p_m are each n×1 (one noise channel per state), combined via hcat.
    p_um(t, x, dp) = zeros(SMatrix{12, 1, eltype(x), 12})
    p_m( t, x, dp) = zeros(SMatrix{12, 1, eltype(x), 12})
    p(   t, x, dp) = zeros(SMatrix{12, 2, eltype(x), 24})

    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    # Uncertain components — placeholder magnitudes, fully typed to match state
    function Λμ_um(t, x, dp)
        T = eltype(x)
        return T(1e-3) * (one(T) + norm(SVector{3,T}(x[1], x[2], x[3])))
    end
    function Λμ_m(t, x, dp)
        T = eltype(x)
        return T(5e-2) * (one(T) + norm(SVector{3,T}(x[4], x[5], x[6])))
    end
    Λμ(t, x, dp) = SVector{2,eltype(x)}(Λμ_um(t, x, dp), Λμ_m(t, x, dp))

    # Λσ follows the same n×d convention as p
    Λσ_um(t, x, dp) = zeros(SMatrix{12, 1, eltype(x), 12})
    Λσ_m( t, x, dp) = zeros(SMatrix{12, 1, eltype(x), 12})
    Λσ(   t, x, dp) = zeros(SMatrix{12, 2, eltype(x), 24})

    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial distributions — nominal starts at x0, true system starts nearby
    x0_vec    = Vector(x0)
    perturbed = x0_vec .+ [0.1, 0.1, -0.1, zeros(9)...]
    nominal_ξ₀ = MvNormal(x0_vec,   1e-4 * I(12))
    true_ξ₀    = MvNormal(perturbed, 1e-4 * I(12))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    nominal_system = nom_sys(system_dimensions, nominal_components, initial_distributions)
    true_system    = true_sys(system_dimensions, nominal_components,
                               uncertain_components, initial_distributions)

    ω        = 50.0
    Tₛ       = 10 * Δₜ
    λₛ       = 100.0
    L1params = drac_params(ω, Tₛ, λₛ)

    return (
        simulation_parameters = simulation_parameters,
        nominal_system        = nominal_system,
        true_system           = true_system,
        L1params              = L1params,
        system_dimensions     = system_dimensions,
    )
end

###################################################################
## MAIN
###################################################################

function main(; Ntraj=Int(1e1), max_GPUs=0,
               systems=[:nominal_sys, :true_sys, :L1_sys])
    @info "Warmup run for JIT compilation"
    println("=====================================")
    warmup_setup = setup_system(; Ntraj=10)
    run_simulations(warmup_setup; max_GPUs=max_GPUs, systems=systems)

    println("=====================================")
    @info "Complete run for Ntraj=$Ntraj"
    println("=====================================")
    setup     = setup_system(; Ntraj=Ntraj)
    solutions = run_simulations(setup; max_GPUs=max_GPUs, systems=systems)
    return setup, solutions
end

## SOLVING
setup, solutions = main();
nominal_sol, true_sol, L1_sol = solutions;

###################################################################
## DATA LOGGING
###################################################################

function log_state_results(setup, solutions; path=joinpath(@__DIR__, "sol_logs"))
    state_logging(setup.system_dimensions;
        sol_nominal = solutions.nominal_sol,
        sol_true    = solutions.true_sol,
        sol_L1      = solutions.L1_sol,
        path        = path)
end

log_state_results(setup, solutions)

###################################################################
## PLOTS
###################################################################
include("plotting_utils.jl")

function generate_state_plots(; path=joinpath(@__DIR__, "sol_logs"), max_traj=50)
    nom = load(joinpath(path, "states_nominal.jld2"))
    tru = load(joinpath(path, "states_true.jld2"))
    L1  = load(joinpath(path, "states_L1.jld2"))

    fig = plot_results(nom, tru, L1; max_traj=max_traj)
    savefig(fig, joinpath(@__DIR__, "states_plot.png"))
    @info "Saved states_plot.png"
    return fig
end

generate_state_plots(; max_traj=50)
