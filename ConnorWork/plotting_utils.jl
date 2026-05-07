# plotting_utils.jl
# Generic state trajectory plotting for n-state systems.
# Works with any state dimension; state labels are auto-generated if not supplied.

using Plots
using LaTeXStrings
using StaticArrays
using JLD2

# State labels for the 12-state quadrotor (used as default)
const QUAD_STATE_LABELS = [
    "px", "py", "pz",
    "vx", "vy", "vz",
    "φ",  "θ",  "ψ",
    "p",  "q",  "r",
]

# extract_state_matrix: pull state component `i` into a T×n_show matrix.
# data["u"] is Vector{Vector{SVector{n,Float64}}} — trajectories × timepoints.
function extract_state_matrix(data; state_component::Int, max_traj=50)
    u      = data["u"]
    n_show = min(length(u), max_traj)
    return reduce(hcat, [getindex.(u[j], state_component) for j in 1:n_show])
end

# build_trajectory_panel: overlay individual trajectories for all three systems.
function build_trajectory_panel(nom, tru, L1; state_component::Int, max_traj=50,
                                  lw=0.8, lalpha=0.15, title_str="")
    t_nom = nom["t"];  t_tru = tru["t"];  t_L1 = L1["t"]

    nom_flat = extract_state_matrix(nom; state_component, max_traj)
    tru_flat = extract_state_matrix(tru; state_component, max_traj)
    L1_flat  = extract_state_matrix(L1;  state_component, max_traj)

    p = plot(t_nom, nom_flat, color=13, lw=lw, linealpha=lalpha, label=false, legend=false)
    plot!(p, t_tru, tru_flat, color=7,  lw=lw, linealpha=lalpha, label=false)
    plot!(p, t_L1,  L1_flat,  color=25, lw=lw, linealpha=lalpha, label=false)
    plot!(p, title=title_str, xlabel="t")
    return p
end

# build_summary_panel: ensemble mean ± 1 std-dev ribbon for one state component.
function build_summary_panel(nom, tru, L1; state_component::Int,
                               lw=1, fillalpha=0.2, title_str="")
    nom_mean = getindex.(nom["mean"], state_component)
    nom_std  = sqrt.(getindex.(nom["var"], state_component))

    tru_mean = getindex.(tru["mean"], state_component)
    tru_std  = sqrt.(getindex.(tru["var"], state_component))

    L1_mean  = getindex.(L1["mean"],  state_component)
    L1_std   = sqrt.(getindex.(L1["var"],  state_component))

    p = plot(nom["t"], nom_mean, ribbon=nom_std, color=13, lw=lw, fillalpha=fillalpha, label=false)
    plot!(p, tru["t"], tru_mean, ribbon=tru_std, color=7,  lw=lw, fillalpha=fillalpha, label=false)
    plot!(p, L1["t"],  L1_mean,  ribbon=L1_std,  color=25, lw=lw, fillalpha=fillalpha, label=false)
    plot!(p, xlabel="t", title=title_str)
    return p
end

# plot_results: build a 2×n grid (trajectory cloud top, summary ribbon bottom).
# n_states is inferred from the data if not provided.
# labels defaults to QUAD_STATE_LABELS for 12-state systems, else "X1", "X2", …
function plot_results(nom, tru, L1; max_traj=50,
                       labels=nothing, fig_width=300, fig_height=400)
    n_states = length(nom["mean"][1])

    if labels === nothing
        labels = (n_states == 12) ? QUAD_STATE_LABELS :
                  ["X$i" for i in 1:n_states]
    end

    traj_panels    = [build_trajectory_panel(nom, tru, L1;
                           state_component=i, max_traj=max_traj,
                           title_str=labels[i]) for i in 1:n_states]
    summary_panels = [build_summary_panel(nom, tru, L1;
                           state_component=i,
                           title_str=labels[i]) for i in 1:n_states]

    all_panels = vcat(traj_panels, summary_panels)

    # Determine a reasonable grid layout: 2 rows (cloud / summary) × n_states columns
    l = @layout [grid(1, n_states); grid(1, n_states)]
    return plot(all_panels..., layout=l,
                size=(fig_width * n_states, fig_height * 2))
end
