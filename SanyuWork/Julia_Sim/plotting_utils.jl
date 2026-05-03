# plotting_utils.jl: State trajectory plotting for the 12-state quadrotor example.
#
# Saved data structure:
#   data["t"]    = time vector
#   data["u"]    = Vector of trajectories, each trajectory is a Vector of state vectors
#   data["mean"] = mean state at each time
#   data["var"]  = state variance at each time

using Plots

const QUAD_STATE_LABELS = [
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "roll",
    "pitch",
    "yaw",
    "roll rate",
    "pitch rate",
    "yaw rate",
]

function _state_count(data)
    return length(first(first(data["u"])))
end

function _state_label(i)
    return i <= length(QUAD_STATE_LABELS) ? QUAD_STATE_LABELS[i] : "state $i"
end

function _trajectory_matrix(data, component; max_traj=1000)
    trajectories = data["u"]
    n_show = min(length(trajectories), max_traj)
    return reduce(hcat, [getindex.(trajectories[i], component) for i in 1:n_show])
end

function build_trajectory_cloud_panel(nom, tru, L1; state_component::Int,
                                      max_traj=1000, show_legend=false,
                                      linealpha=0.05)
    title_str = _state_label(state_component)

    nom_cloud = _trajectory_matrix(nom, state_component; max_traj=max_traj)
    tru_cloud = _trajectory_matrix(tru, state_component; max_traj=max_traj)
    L1_cloud = _trajectory_matrix(L1, state_component; max_traj=max_traj)

    legend_setting = show_legend ? :topright : false
    p = plot(
        nom["t"],
        nom_cloud;
        color=:dodgerblue3,
        lw=0.8,
        linealpha=linealpha,
        label=false,
        legend=legend_setting,
        title=title_str,
        xlabel="t",
    )
    plot!(
        p,
        tru["t"],
        tru_cloud;
        color=:firebrick3,
        lw=0.8,
        linealpha=linealpha,
        label=false,
    )
    plot!(
        p,
        L1["t"],
        L1_cloud;
        color=:seagreen4,
        lw=0.8,
        linealpha=linealpha,
        label=false,
    )

    if show_legend
        plot!(p, [NaN], [NaN]; color=:dodgerblue3, lw=2, label="nominal")
        plot!(p, [NaN], [NaN]; color=:firebrick3, lw=2, label="true disturbed")
        plot!(p, [NaN], [NaN]; color=:seagreen4, lw=2, label="L1-DRAC")
    end

    return p
end

function plot_position_results(nom, tru, L1; max_traj=1000)
    panels = [
        build_trajectory_cloud_panel(nom, tru, L1; state_component=1, max_traj=max_traj, show_legend=true),
        build_trajectory_cloud_panel(nom, tru, L1; state_component=2, max_traj=max_traj),
        build_trajectory_cloud_panel(nom, tru, L1; state_component=3, max_traj=max_traj),
    ]
    return plot(panels..., layout=(3, 1), size=(900, 900), plot_title="Quadrotor position trajectory clouds")
end

function plot_results(nom, tru, L1; max_traj=500)
    nstates = min(_state_count(nom), length(QUAD_STATE_LABELS))
    panels = [
        build_trajectory_cloud_panel(nom, tru, L1; state_component=i,
                                     max_traj=max_traj, show_legend=(i == 1))
        for i in 1:nstates
    ]
    return plot(panels..., layout=(4, 3), size=(1500, 1200), plot_title="Quadrotor trajectory clouds")
end
