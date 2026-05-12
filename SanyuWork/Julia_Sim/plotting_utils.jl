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
        color=:black,
        lw=1.0,
        linealpha=max(linealpha, 0.12),
        label=false,
    )

    plot!(p, nom["t"], getindex.(nom["mean"], state_component);
          color=:dodgerblue3, lw=2.5, label=false)
    plot!(p, tru["t"], getindex.(tru["mean"], state_component);
          color=:firebrick3, lw=2.5, label=false)
    plot!(p, L1["t"], getindex.(L1["mean"], state_component);
          color=:black, lw=2.5, label=false)

    if show_legend
        plot!(p, [NaN], [NaN]; color=:dodgerblue3, lw=2, label="nominal")
        plot!(p, [NaN], [NaN]; color=:firebrick3, lw=2, label="true disturbed")
        plot!(p, [NaN], [NaN]; color=:black, lw=2, label="L1-DRAC")
    end

    return p
end

function build_named_trajectory_cloud_panel(data1, data2, data3; state_component::Int,
                                            labels=("LQR", "iLQR", "LQR + L1-DRAC"),
                                            colors=(:darkorange3, :dodgerblue3, :seagreen4),
                                            max_traj=1000, show_legend=false,
                                            linealpha=0.06)
    title_str = _state_label(state_component)

    cloud1 = _trajectory_matrix(data1, state_component; max_traj=max_traj)
    cloud2 = _trajectory_matrix(data2, state_component; max_traj=max_traj)
    cloud3 = _trajectory_matrix(data3, state_component; max_traj=max_traj)

    legend_setting = show_legend ? :topright : false
    p = plot(
        data1["t"],
        cloud1;
        color=colors[1],
        lw=0.8,
        linealpha=linealpha,
        label=false,
        legend=legend_setting,
        title=title_str,
        xlabel="t",
    )
    plot!(
        p,
        data2["t"],
        cloud2;
        color=colors[2],
        lw=0.8,
        linealpha=linealpha,
        label=false,
    )
    plot!(
        p,
        data3["t"],
        cloud3;
        color=colors[3],
        lw=0.8,
        linealpha=linealpha,
        label=false,
    )

    if show_legend
        plot!(p, [NaN], [NaN]; color=colors[1], lw=2, label=labels[1])
        plot!(p, [NaN], [NaN]; color=colors[2], lw=2, label=labels[2])
        plot!(p, [NaN], [NaN]; color=colors[3], lw=2, label=labels[3])
    end

    return p
end

function build_nominal_trajectory_panel(nom; state_component::Int, x_goal=nothing,
                                        max_traj=1000, show_legend=false,
                                        linealpha=0.25)
    title_str = _state_label(state_component)
    nom_cloud = _trajectory_matrix(nom, state_component; max_traj=max_traj)

    legend_setting = show_legend ? :topright : false
    p = plot(
        nom["t"],
        nom_cloud;
        color=:dodgerblue3,
        lw=1.0,
        linealpha=linealpha,
        label=false,
        legend=legend_setting,
        title=title_str,
        xlabel="t",
    )

    if x_goal !== nothing
        hline!(p, [x_goal[state_component]]; color=:black, lw=2, ls=:dash, label=false)
    end

    if show_legend
        plot!(p, [NaN], [NaN]; color=:dodgerblue3, lw=2, label="nominal trajectories")
        if x_goal !== nothing
            plot!(p, [NaN], [NaN]; color=:black, lw=2, ls=:dash, label="goal")
        end
    end

    return p
end

function build_baseline_adaptive_panel(nom, L1; state_component::Int,
                                       max_traj=1000, show_legend=false,
                                       linealpha=0.06)
    title_str = _state_label(state_component)

    nom_cloud = _trajectory_matrix(nom, state_component; max_traj=max_traj)
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
        L1["t"],
        L1_cloud;
        color=:seagreen4,
        lw=0.8,
        linealpha=linealpha,
        label=false,
    )

    if show_legend
        plot!(p, [NaN], [NaN]; color=:dodgerblue3, lw=2, label="nominal")
        plot!(p, [NaN], [NaN]; color=:seagreen4, lw=2, label="L1-DRAC")
    end

    return p
end

function plot_nominal_position_results(nom; x_goal=nothing, max_traj=1000)
    panels = [
        build_nominal_trajectory_panel(nom; state_component=1, x_goal=x_goal,
                                       max_traj=max_traj, show_legend=true),
        build_nominal_trajectory_panel(nom; state_component=2, x_goal=x_goal,
                                       max_traj=max_traj),
        build_nominal_trajectory_panel(nom; state_component=3, x_goal=x_goal,
                                       max_traj=max_traj),
    ]
    return plot(panels..., layout=(3, 1), size=(900, 900),
                plot_title="Nominal quadrotor position trajectories")
end

function plot_nominal_xyz_paths(nom; x_goal=nothing, max_traj=1000)
    trajectories = nom["u"]
    n_show = min(length(trajectories), max_traj)

    p = plot(
        legend=:topright,
        xlabel="x",
        ylabel="y",
        zlabel="z",
        title="Nominal quadrotor paths to shared goal",
        size=(900, 750),
    )

    for i in 1:n_show
        xs = getindex.(trajectories[i], 1)
        ys = getindex.(trajectories[i], 2)
        zs = getindex.(trajectories[i], 3)
        plot!(p, xs, ys, zs; color=:dodgerblue3, lw=1.0, linealpha=0.25, label=false)
    end

    if x_goal !== nothing
        scatter!(p, [x_goal[1]], [x_goal[2]], [x_goal[3]];
                 color=:black, markersize=5, label="goal")
    end

    plot!(p, [NaN], [NaN], [NaN]; color=:dodgerblue3, lw=2, label="nominal trajectories")
    return p
end

function plot_position_results(nom, tru, L1; max_traj=1000,
                               plot_title="Quadrotor position trajectory clouds")
    panels = [
        build_trajectory_cloud_panel(nom, tru, L1; state_component=1, max_traj=max_traj, show_legend=true),
        build_trajectory_cloud_panel(nom, tru, L1; state_component=2, max_traj=max_traj),
        build_trajectory_cloud_panel(nom, tru, L1; state_component=3, max_traj=max_traj),
    ]
    return plot(panels..., layout=(3, 1), size=(900, 900), plot_title=plot_title)
end

function plot_baseline_adaptive_position_results(nom, L1; max_traj=1000,
                                                 plot_title="Nominal vs L1-DRAC position trajectory clouds")
    panels = [
        build_baseline_adaptive_panel(nom, L1; state_component=1, max_traj=max_traj, show_legend=true),
        build_baseline_adaptive_panel(nom, L1; state_component=2, max_traj=max_traj),
        build_baseline_adaptive_panel(nom, L1; state_component=3, max_traj=max_traj),
    ]
    return plot(panels..., layout=(3, 1), size=(900, 900), plot_title=plot_title)
end

function plot_results(nom, tru, L1; max_traj=500,
                      plot_title="Quadrotor trajectory clouds")
    nstates = min(_state_count(nom), length(QUAD_STATE_LABELS))
    panels = [
        build_trajectory_cloud_panel(nom, tru, L1; state_component=i,
                                     max_traj=max_traj, show_legend=(i == 1))
        for i in 1:nstates
    ]
    return plot(panels..., layout=(4, 3), size=(1500, 1200), plot_title=plot_title)
end

function plot_controller_comparison_position_results(lqr, ilqr, lqr_l1; max_traj=1000,
                                                     plot_title="LQR vs iLQR vs LQR + L1-DRAC position trajectory clouds")
    panels = [
        build_named_trajectory_cloud_panel(lqr, ilqr, lqr_l1; state_component=1,
                                           max_traj=max_traj, show_legend=true),
        build_named_trajectory_cloud_panel(lqr, ilqr, lqr_l1; state_component=2,
                                           max_traj=max_traj),
        build_named_trajectory_cloud_panel(lqr, ilqr, lqr_l1; state_component=3,
                                           max_traj=max_traj),
    ]
    return plot(panels..., layout=(3, 1), size=(900, 900), plot_title=plot_title)
end

function plot_controller_comparison_results(lqr, ilqr, lqr_l1; max_traj=500,
                                            plot_title="LQR vs iLQR vs LQR + L1-DRAC trajectory clouds")
    nstates = min(_state_count(lqr), length(QUAD_STATE_LABELS))
    panels = [
        build_named_trajectory_cloud_panel(lqr, ilqr, lqr_l1; state_component=i,
                                           max_traj=max_traj, show_legend=(i == 1))
        for i in 1:nstates
    ]
    return plot(panels..., layout=(4, 3), size=(1500, 1200), plot_title=plot_title)
end

function plot_controller_resource_metrics(metrics;
                                          plot_title="Controller computational resource comparison")
    rows = [metrics.lqr, metrics.ilqr, metrics.lqr_l1]
    labels = [row.controller for row in rows]
    wall_seconds = [row.wall_seconds for row in rows]
    seconds_per_traj = [row.seconds_per_trajectory for row in rows]
    allocated_mb = [row.allocated_mb for row in rows]

    p_time = bar(
        labels,
        wall_seconds;
        ylabel="seconds",
        title="wall time",
        label=false,
        color=[:darkorange3 :dodgerblue3 :seagreen4],
        xrotation=20,
    )
    p_per_traj = bar(
        labels,
        seconds_per_traj;
        ylabel="seconds / trajectory",
        title="time per trajectory",
        label=false,
        color=[:darkorange3 :dodgerblue3 :seagreen4],
        xrotation=20,
    )
    p_memory = bar(
        labels,
        allocated_mb;
        ylabel="MB",
        title="allocated memory",
        label=false,
        color=[:darkorange3 :dodgerblue3 :seagreen4],
        xrotation=20,
    )

    return plot(p_time, p_per_traj, p_memory; layout=(1, 3), size=(1200, 400),
                plot_title=plot_title)
end

function plot_baseline_adaptive_results(nom, L1; max_traj=500,
                                        plot_title="Nominal vs L1-DRAC trajectory clouds")
    nstates = min(_state_count(nom), length(QUAD_STATE_LABELS))
    panels = [
        build_baseline_adaptive_panel(nom, L1; state_component=i,
                                      max_traj=max_traj, show_legend=(i == 1))
        for i in 1:nstates
    ]
    return plot(panels..., layout=(4, 3), size=(1500, 1200), plot_title=plot_title)
end
