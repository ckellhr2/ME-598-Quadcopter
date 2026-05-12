include("quadcopter_ilqr_adaptive.jl")

using Statistics

const OBSTACLE_AVOIDANCE_OUTPUT_DIR = joinpath(@__DIR__, "Obstacle Avoidance")
const CORRIDOR_LOG_ROOT = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "quad_corridor_sol_logs")

function clamp01(x)
    return clamp(x, 0.0, 1.0)
end

function s_curve_reference(t; t_final=14.0)
    s = clamp01(t / t_final)
    dsdt = 0.0 <= t <= t_final ? 1.0 / t_final : 0.0

    x = -1.5 + 4.2 * s
    y = -1.25 * cos(2.5 * pi * s)
    z = 1.05 + 0.35 * s + 0.12 * sin(2.0 * pi * s)

    vx = 4.2 * dsdt
    vy = 1.25 * 2.5 * pi * sin(2.5 * pi * s) * dsdt
    vz = (0.35 + 0.12 * 2.0 * pi * cos(2.0 * pi * s)) * dsdt

    ref = zeros(NX_QUAD)
    ref[1:6] .= [x, y, z, vx, vy, vz]
    return ref
end

function _segment_reference(t, waypoints, t_final)
    nseg = length(waypoints) - 1
    tau = clamp(t, 0.0, t_final)
    seg_time = t_final / nseg
    idx = min(Int(floor(tau / seg_time)) + 1, nseg)
    local_s = clamp01((tau - (idx - 1) * seg_time) / seg_time)

    p0 = waypoints[idx]
    p1 = waypoints[idx + 1]
    smooth_s = local_s^2 * (3.0 - 2.0 * local_s)
    smooth_ds = 6.0 * local_s * (1.0 - local_s) / seg_time

    pos = p0 .+ smooth_s .* (p1 .- p0)
    vel = smooth_ds .* (p1 .- p0)

    ref = zeros(NX_QUAD)
    ref[1:3] .= pos
    ref[4:6] .= vel
    return ref
end

function sharp_turn_reference(t; t_final=14.0)
    waypoints = [
        [-1.5, -1.5, 1.05],
        [-0.9,  1.35, 1.25],
        [ 0.35, -1.25, 1.45],
        [ 1.45,  1.15, 1.10],
        [ 2.45, -0.15, 1.40],
    ]
    return _segment_reference(t, waypoints, t_final)
end

function corridor_reference(track, t; t_final=14.0)
    track == :s_curve && return s_curve_reference(t; t_final=t_final)
    track == :sharp_turns && return sharp_turn_reference(t; t_final=t_final)
    error("Unknown corridor track: $track")
end

function corridor_obstacles(track)
    if track == :s_curve
        return []
    elseif track == :sharp_turns
        return []
    end
    error("Unknown corridor track: $track")
end

function scenario_profiles(scenario)
    if scenario == :baseline
        return (
            wind_profile=NO_WIND_PROFILE,
            fault_profile=DEFAULT_FAULT_PROFILE,
            aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
            plant_mismatch_enabled=false,
            title="Baseline: Brownian and aleatoric uncertainty only",
        )
    elseif scenario == :wind_gust
        return (
            wind_profile=WIND_GUST_PROFILE,
            fault_profile=DEFAULT_FAULT_PROFILE,
            aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
            plant_mismatch_enabled=true,
            title="Strong wind plus sudden gust",
        )
    elseif scenario == :propeller_failure
        return (
            wind_profile=NO_WIND_PROFILE,
            fault_profile=PROPELLER_FAILURE_PROFILE,
            aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
            plant_mismatch_enabled=true,
            title="Propeller failure: rotor 1 at 70%",
        )
    elseif scenario == :variable_thrust
        return (
            wind_profile=NO_WIND_PROFILE,
            fault_profile=VARIABLE_THRUST_PROFILE,
            aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
            plant_mismatch_enabled=true,
            title="Variable rotor thrust effectiveness",
        )
    end
    error("Unknown corridor scenario: $scenario")
end

function _start_offsets(ref0)
    offset = zeros(NX_QUAD)
    offset[1:3] .= ref0[1:3] .- [0.0, 0.0, 1.0]
    return Dict(:nominal_sys => offset, :true_sys => offset, :L1_sys => offset)
end

function _trajectory_positions(traj)
    return reduce(hcat, [state[1:3] for state in traj])
end

function _mean_positions(data)
    return reduce(hcat, [state[1:3] for state in data["mean"]])
end

function _reference_positions(tvec, ref_func)
    return reduce(hcat, [ref_func(t)[1:3] for t in tvec])
end

function clearance_to_obstacles(pos, obstacles)
    isempty(obstacles) && return Inf
    clearances = [norm(pos .- obs.center) - obs.radius for obs in obstacles]
    return minimum(clearances)
end

function tracking_metrics(data, ref_func, obstacles)
    tvec = data["t"]
    rmses = Float64[]
    max_errors = Float64[]
    min_clearances = Float64[]
    collisions = 0

    for traj in data["u"]
        errors = Float64[]
        clearances = Float64[]
        for (k, state) in enumerate(traj)
            pos = state[1:3]
            ref = ref_func(tvec[k])[1:3]
            push!(errors, norm(pos .- ref))
            push!(clearances, clearance_to_obstacles(pos, obstacles))
        end
        push!(rmses, sqrt(mean(abs2, errors)))
        push!(max_errors, maximum(errors))
        push!(min_clearances, minimum(clearances))
        collisions += minimum(clearances) < 0.0 ? 1 : 0
    end

    return (
        rmse = mean(rmses),
        max_error = mean(max_errors),
        worst_clearance = minimum(min_clearances),
        mean_clearance = mean(min_clearances),
        collision_rate = collisions / length(data["u"]),
    )
end

function save_corridor_metrics(metrics; output_prefix)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    path = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "$(output_prefix)_metrics.csv")
    open(path, "w") do io
        println(io, "system,rmse,max_error,worst_clearance,mean_clearance,collision_rate")
        for name in [:nominal, :true_disturbed, :L1_DRAC]
            row = metrics[name]
            println(io, join((String(name), row.rmse, row.max_error, row.worst_clearance,
                              row.mean_clearance, row.collision_rate), ","))
        end
    end
    @info "Saved $(basename(path))"
    return path
end

function _plot_obstacles_2d!(p, obstacles)
    theta = range(0, 2pi; length=80)
    for obs in obstacles
        xs = obs.center[1] .+ obs.radius .* cos.(theta)
        ys = obs.center[2] .+ obs.radius .* sin.(theta)
        plot!(p, xs, ys; color=:gray35, fill=(0, 0.18, :gray35), lw=1.5, label=false)
    end
end

function plot_corridor_paths(nom, tru, L1, ref_func, obstacles; output_prefix, plot_title)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    tvec = nom["t"]
    ref = _reference_positions(tvec, ref_func)
    nom_mean = _mean_positions(nom)
    tru_mean = _mean_positions(tru)
    L1_mean = _mean_positions(L1)

    p = plot(ref[1, :], ref[2, :]; color=:black, ls=:dash, lw=2.5,
             label="reference", xlabel="x", ylabel="y",
             title=plot_title, size=(900, 700), aspect_ratio=:equal)
    _plot_obstacles_2d!(p, obstacles)
    plot!(p, nom_mean[1, :], nom_mean[2, :]; color=:dodgerblue3, lw=2.5, label="nominal")
    plot!(p, tru_mean[1, :], tru_mean[2, :]; color=:firebrick3, lw=2.5, label="true disturbed")
    plot!(p, L1_mean[1, :], L1_mean[2, :]; color=:black, lw=3.0, label="L1-DRAC")

    filename = "$(output_prefix)_xy_corridor_plot.png"
    savefig(p, joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, filename))
    @info "Saved $filename"
    return p
end

function make_corridor_gif(nom, tru, L1, ref_func, obstacles; output_prefix, plot_title, nframes=90)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    tvec = nom["t"]
    idxs = unique(round.(Int, range(1, length(tvec); length=min(nframes, length(tvec)))))
    ref = _reference_positions(tvec, ref_func)
    nom_mean = _mean_positions(nom)
    tru_mean = _mean_positions(tru)
    L1_mean = _mean_positions(L1)

    anim = @animate for idx in idxs
        p = plot(ref[1, :], ref[2, :]; color=:black, ls=:dash, lw=2,
                 label="reference", xlabel="x", ylabel="y", title=plot_title,
                 aspect_ratio=:equal, xlim=(-2.0, 3.0), ylim=(-2.0, 1.8),
                 size=(700, 550))
        _plot_obstacles_2d!(p, obstacles)
        plot!(p, nom_mean[1, 1:idx], nom_mean[2, 1:idx]; color=:dodgerblue3, lw=2, label="nominal")
        plot!(p, tru_mean[1, 1:idx], tru_mean[2, 1:idx]; color=:firebrick3, lw=2, label="true disturbed")
        plot!(p, L1_mean[1, 1:idx], L1_mean[2, 1:idx]; color=:black, lw=2.5, label="L1-DRAC")
        scatter!(p, [nom_mean[1, idx]], [nom_mean[2, idx]]; color=:dodgerblue3, ms=4, label=false)
        scatter!(p, [tru_mean[1, idx]], [tru_mean[2, idx]]; color=:firebrick3, ms=4, label=false)
        scatter!(p, [L1_mean[1, idx]], [L1_mean[2, idx]]; color=:black, ms=4, label=false)
    end

    filename = "$(output_prefix)_mean_paths.gif"
    gif(anim, joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, filename), fps=15)
    @info "Saved $filename"
    return filename
end

function analyze_corridor_run(; log_path, ref_func, obstacles, output_prefix, plot_title, make_gif=true)
    nom = load(joinpath(log_path, "states_nominal.jld2"))
    tru = load(joinpath(log_path, "states_true.jld2"))
    L1 = load(joinpath(log_path, "states_L1.jld2"))

    metrics = (
        nominal = tracking_metrics(nom, ref_func, obstacles),
        true_disturbed = tracking_metrics(tru, ref_func, obstacles),
        L1_DRAC = tracking_metrics(L1, ref_func, obstacles),
    )
    save_corridor_metrics(metrics; output_prefix=output_prefix)
    path_plot = plot_corridor_paths(nom, tru, L1, ref_func, obstacles;
                                    output_prefix=output_prefix, plot_title=plot_title)
    position_plot, state_plot = generate_three_system_comparison_plots(;
        path=log_path,
        output_prefix=output_prefix,
        plot_title=plot_title,
        output_dir=OBSTACLE_AVOIDANCE_OUTPUT_DIR,
    )
    gif_file = make_gif ?
        make_corridor_gif(nom, tru, L1, ref_func, obstacles;
                          output_prefix=output_prefix, plot_title=plot_title) :
        nothing

    return (metrics=metrics, path_plot=path_plot, position_plot=position_plot,
            state_plot=state_plot, gif_file=gif_file)
end

function run_corridor_experiment(; track=:s_curve,
                                 scenario=:baseline,
                                 Ntraj=100,
                                 t_final=14.0,
                                 max_GPUs=0,
                                 dt=1e-2,
                                 save_stride=5,
                                 ilqr_horizon=80,
                                 ilqr_max_iter=5,
                                 reference_lookahead=0.35,
                                 output_tag="$(track)_$(scenario)",
                                 make_gif=true,
                                 quiet=true)
    profiles = scenario_profiles(scenario)
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    obstacles = corridor_obstacles(track)
    ref0 = ref_func(0.0)

    setup, solutions = main(;
        Ntraj=Ntraj,
        max_GPUs=max_GPUs,
        systems=[:nominal_sys, :true_sys, :L1_sys],
        t_final=t_final,
        dt=dt,
        save_stride=save_stride,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        initial_mean_offsets=_start_offsets(ref0),
        reference_func=ref_func,
        reference_lookahead=reference_lookahead,
        wind_profile=profiles.wind_profile,
        fault_profile=profiles.fault_profile,
        aero_damage_profile=profiles.aero_damage_profile,
        plant_mismatch_enabled=profiles.plant_mismatch_enabled,
        L1_omega=60.0,
        L1_sample_steps=2,
        L1_predictor_lambda=120.0,
        L1_control_limit=8.0,
        L1_lateral_tilt_gain=45.0,
        L1_lateral_rate_damping=10.0,
        L1_max_tilt_correction=1.10,
        L1_lateral_position_gain=2.50,
        L1_lateral_velocity_gain=3.50,
        quiet=quiet,
    )

    log_path = joinpath(CORRIDOR_LOG_ROOT, output_tag)
    log_state_results(setup, solutions; path=log_path)

    title_track = track == :s_curve ? "S-Curve Corridor" : "Sharp Waypoint Turns"
    plot_title = "$title_track: $(profiles.title)"
    return analyze_corridor_run(;
        log_path=log_path,
        ref_func=ref_func,
        obstacles=obstacles,
        output_prefix="quad_corridor_$(output_tag)",
        plot_title=plot_title,
        make_gif=make_gif,
    )
end

function run_corridor_batch(; tracks=[:s_curve, :sharp_turns],
                            scenarios=[:wind_gust, :propeller_failure, :baseline],
                            Ntraj=100,
                            t_final=14.0,
                            max_GPUs=0,
                            make_gif=true,
                            quiet=true,
                            kwargs...)
    results = Dict{Tuple{Symbol, Symbol}, Any}()
    for track in tracks
        for scenario in scenarios
            @info "Running corridor experiment" track=track scenario=scenario
            results[(track, scenario)] = run_corridor_experiment(;
                kwargs...,
                track=track,
                scenario=scenario,
                Ntraj=Ntraj,
                t_final=t_final,
                max_GPUs=max_GPUs,
                make_gif=make_gif,
                quiet=quiet,
            )
        end
    end
    save_corridor_success_summary(results)
    return results
end

function regenerate_corridor_outputs_from_logs(; tracks=[:s_curve, :sharp_turns],
                                               scenarios=[:wind_gust, :propeller_failure, :variable_thrust, :baseline],
                                               t_final=10.0,
                                               make_gif=true)
    results = Dict{Tuple{Symbol, Symbol}, Any}()

    for track in tracks
        for scenario in scenarios
            output_tag = "$(track)_$(scenario)"
            log_path = joinpath(CORRIDOR_LOG_ROOT, output_tag)
            isdir(log_path) || begin
                @warn "Skipping missing corridor log folder" log_path=log_path
                continue
            end

            profiles = scenario_profiles(scenario)
            ref_func = t -> corridor_reference(track, t; t_final=t_final)
            obstacles = corridor_obstacles(track)
            title_track = track == :s_curve ? "S-Curve Corridor" : "Sharp Waypoint Turns"
            plot_title = "$title_track: $(profiles.title)"

            @info "Regenerating corridor outputs from logs" track=track scenario=scenario
            results[(track, scenario)] = analyze_corridor_run(;
                log_path=log_path,
                ref_func=ref_func,
                obstacles=obstacles,
                output_prefix="quad_corridor_$(output_tag)",
                plot_title=plot_title,
                make_gif=make_gif,
            )
        end
    end

    isempty(results) || save_corridor_success_summary(results)
    return results
end

function save_corridor_success_summary(results; filename="quad_corridor_success_summary.png")
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)

    case_labels = String[]
    true_success = Float64[]
    L1_success = Float64[]

    for key in sort(collect(keys(results)); by=x -> string(x[1]) * "_" * string(x[2]))
        track, scenario = key
        push!(case_labels, "$(track)\n$(scenario)")
        metrics = results[key].metrics
        push!(true_success, 100.0 * (1.0 - metrics.true_disturbed.collision_rate))
        push!(L1_success, 100.0 * (1.0 - metrics.L1_DRAC.collision_rate))
    end

    p = bar(
        case_labels,
        [true_success L1_success];
        label=["true disturbed iLQR" "L1-DRAC"],
        color=[:firebrick3 :black],
        ylabel="success rate (%)",
        ylim=(0, 105),
        title="Obstacle avoidance success rate",
        xrotation=25,
        size=(1100, 550),
        legend=:bottomright,
    )

    savefig(p, joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, filename))
    @info "Saved $filename"
    return p
end
