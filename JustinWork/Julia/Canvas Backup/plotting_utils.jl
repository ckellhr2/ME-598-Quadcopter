# plotting_utils.jl: State trajectory plotting for examples/ex1 (double integrator, 2 states)
# Extracts each state component (X1, X2) into flat T x Ntraj matrices for plotting.

using Plots
using LaTeXStrings
using StaticArrays
using JLD2

# Trajectory data arrives as named tuples with .u = Vector{Vector{SVector{2,Float64}}}
# Each state component is extracted separately into a flat Float64 matrix.

# extract_state_matrix: Extract the chosen state component into a T x n_show matrix
# n_show = min(Ntraj, max_traj) — user-chosen cap on trajectories to display
# columns = trajectories, rows = timepoints
function extract_state_matrix(data; state_component::Int, max_traj=500)
    u = data["u"]
    n_show = min(length(u), max_traj)
    return reduce(hcat, [getindex.(u[i], state_component) for i in 1:n_show])
end

# build_trajectory_panel: Plot individual trajectories for one state component
# Overlays all three systems (nom, tru, L1) on a single panel
# Colors: nominal=13, true=7, L1=25 (palette indices), low alpha for cloud effect
function build_trajectory_panel(nom, tru, L1; state_component::Int, max_traj=500,
                                 lw=1, lalpha=0.1, title_str="")
    t_nom = nom["t"]
    t_tru = tru["t"]
    t_L1  = L1["t"]

    nom_flat = extract_state_matrix(nom; state_component=state_component, max_traj=max_traj)
    tru_flat = extract_state_matrix(tru; state_component=state_component, max_traj=max_traj)
    L1_flat  = extract_state_matrix(L1;  state_component=state_component, max_traj=max_traj)

    p = plot(t_nom, nom_flat, color=13, lw=lw, linealpha=lalpha, label=false, legend=false)
    plot!(p, t_tru, tru_flat, color=7,  lw=lw, linealpha=lalpha, label=false)
    plot!(p, t_L1,  L1_flat,  color=25, lw=lw, linealpha=lalpha, label=false)

    plot!(p, title=title_str)
    return p
end

# build_summary_panel: Plot ensemble mean with variance ribbon for one state component
# Same data access pattern as build_trajectory_panel — Dict from JLD2 load().
# data["mean"] is Vector{SVector{2,Float64}} length T — getindex extracts scalar component.
# data["var"] stores VARIANCE — sqrt gives std dev for ribbon half-width (mean ± 1 std dev).
function build_summary_panel(nom, tru, L1; state_component::Int,
                              lw=1, fillalpha=0.2, title_str="")
    nom_mean = getindex.(nom["mean"], state_component)
    nom_std  = sqrt.(getindex.(nom["var"], state_component))

    tru_mean = getindex.(tru["mean"], state_component)
    tru_std  = sqrt.(getindex.(tru["var"], state_component))

    L1_mean = getindex.(L1["mean"], state_component)
    L1_std  = sqrt.(getindex.(L1["var"], state_component))

    p = plot(nom["t"], nom_mean, ribbon=nom_std, color=13, lw=lw, fillalpha=fillalpha, label=false)
    plot!(p, tru["t"], tru_mean, ribbon=tru_std, color=7,  lw=lw, fillalpha=fillalpha, label=false)
    plot!(p, L1["t"],  L1_mean,  ribbon=L1_std,  color=25, lw=lw, fillalpha=fillalpha, label=false)

    plot!(p, xlabel="t", title=title_str)
    return p
end

# plot_results: Main entry point — build 2x2 state trajectory figure
# Top row: individual trajectories (X1 left, X2 right)
# Bottom row: ensemble summary with mean + variance ribbon (X1 left, X2 right)
# Assembled into 2x2 grid via @layout, size 900x900.
function plot_results(nom, tru, L1; max_traj=500)
    p1 = build_trajectory_panel(nom, tru, L1; state_component=1,
             max_traj=max_traj, title_str=L"X_1")
    p2 = build_trajectory_panel(nom, tru, L1; state_component=2,
             max_traj=max_traj, title_str=L"X_2")
    p3 = build_summary_panel(nom, tru, L1; state_component=1)
    p4 = build_summary_panel(nom, tru, L1; state_component=2)

    l = @layout [a b; c d]
    return plot(p1, p2, p3, p4, layout=l, size=(900, 900))
end