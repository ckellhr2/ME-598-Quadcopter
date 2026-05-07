## In the package as examples/ex1/doubleintegrator1D.jl 

using L1DRAC
using CUDA
using LinearAlgebra
using Distributions
using ControlSystemsBase
using StaticArrays
using Plots
using JLD2

###################################################################
## SYSTEM SETUP
###################################################################
function setup_system(; Ntraj=10) # Ntraj = number of trajectories for ensemble sims, default val 10
    # Simulation Parameters
    tspan = (0.0, 5.0)
    Δₜ = 1e-4 # Time step size
    Δ_saveat = 1e2 * Δₜ # Needs to be an integer multiple of Δₜ
    simulation_parameters = sim_params(tspan, Δₜ, Ntraj, Δ_saveat)

    # System Dimensions
    n, m, d = 2, 1, 2
    system_dimensions = sys_dims(n, m, d)

    # Double integrator dynamics
    A = @SMatrix [0.0 1.0; 0.0 0.0]
    B = @SMatrix [0.0; 1.0]

    # Baseline controller via pole placement
    λ = 10.0 # Stability margin
    sys = ss(A, B, SMatrix{2,2}(1.0I), 0.0)
    K = SMatrix{1,2}(place(sys, -λ * ones(2)))
    dp = (; K) # Dynamics params for GPU

    function baseline_input(t, x, dp) # Tracking controller
        r = @SVector [5*sin(t) + 3*cos(2*t), 0.0] # Reference trajectory
        return dp.K * (r - x)
    end

    # Nominal Vector Fields
    f(t, x, dp) = A * x + B * baseline_input(t, x, dp)
    g(t, x, dp) = @SVector [0.0, 1.0]
    g_perp(t, x, dp) = @SVector [1.0, 0.0]

    p_um(t, x, dp) = @SMatrix [0.0 0.0]
    p_m(t, x, dp) = @SMatrix [0.0 0.0]
    p(t, x, dp) = vcat(p_um(t, x, dp), p_m(t, x, dp))

    nominal_components = nominal_vector_fields(f, g, g_perp, p, dp)

    # Uncertain Vector Fields
    Λμ_um(t, x, dp) = 1e-2 * (1 + sin(x[1]))
    Λμ_m(t, x, dp) = 6*(5 + 10*cos(x[2]) + 5*norm(x))
    Λμ(t, x, dp) = @SVector [Λμ_um(t, x, dp), Λμ_m(t, x, dp)]

    Λσ_um(t, x, dp) = @SMatrix [0.0 0.0]
    Λσ_m(t, x, dp) = @SMatrix [0.0 0.0]
    Λσ(t, x, dp) = vcat(Λσ_um(t, x, dp), Λσ_m(t, x, dp))

    uncertain_components = uncertain_vector_fields(Λμ, Λσ)

    # Initial Distributions
    nominal_ξ₀ = MvNormal(20.0 * ones(2), 1e-5 * I(2))
    true_ξ₀ = MvNormal(-2.0 * ones(2), 1e-5 * I(2))
    initial_distributions = init_dist(nominal_ξ₀, true_ξ₀)

    # Define Systems
    nominal_system = nom_sys(system_dimensions, nominal_components, 
                        initial_distributions)
    true_system = true_sys(system_dimensions, nominal_components, 
                        uncertain_components, initial_distributions)

    # L1-DRAC Parameters (PLACEHOLDER values)
    ω = 50.0 # Filter bandwidth
    Tₛ = 10 * Δₜ # Sample time (integer multiple of Δₜ)
    λₛ = 100.0 # Predictor stability
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