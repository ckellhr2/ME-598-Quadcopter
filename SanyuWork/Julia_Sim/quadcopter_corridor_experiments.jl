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

function corridor_track_title(track)
    track == :s_curve && return "S-Curve Corridor"
    track == :sharp_turns && return "Sharp Waypoint Turns"
    track == :straight_line && return "Straight-Line Obstacle Clearance"
    return string(track)
end

function straight_line_reference(t; t_final=14.0)
    waypoints = [
        [-1.5, -1.25, 1.15],
        [ 2.55,  0.00, 1.35],
    ]
    return _segment_reference(t, waypoints, t_final)
end

function corridor_reference(track, t; t_final=14.0)
    track == :s_curve && return s_curve_reference(t; t_final=t_final)
    track == :sharp_turns && return sharp_turn_reference(t; t_final=t_final)
    track == :straight_line && return straight_line_reference(t; t_final=t_final)
    error("Unknown corridor track: $track")
end

function corridor_obstacles(track)
    if track == :s_curve
        return [
            (center=[-0.55, -1.75, 1.20], radius=0.22),
            (center=[ 0.85, -1.75, 1.30], radius=0.24),
            (center=[ 2.10,  0.95, 1.30], radius=0.24),
        ]
    elseif track == :sharp_turns
        return [
            (center=[-0.75, -0.20, 1.20], radius=0.23),
            (center=[ 0.25,  0.45, 1.35], radius=0.24),
            (center=[ 1.55, -0.55, 1.25], radius=0.24),
        ]
    elseif track == :straight_line
        return [
            (center=[0.45, -0.45, 1.25], radius=0.24),
            (center=[1.90,  2.10, 1.30], radius=0.22),
        ]
    end
    error("Unknown corridor track: $track")
end

function _planned_reference_from_points(points, t_final)
    npts = length(points)
    times = collect(range(0.0, t_final; length=npts))

    return function (t)
        tau = clamp(t, 0.0, t_final)
        idx = searchsortedlast(times, tau)
        idx >= npts && begin
            ref = zeros(NX_QUAD)
            ref[1:3] .= points[end]
            return ref
        end
        idx = max(idx, 1)

        t0, t1 = times[idx], times[idx + 1]
        p0, p1 = points[idx], points[idx + 1]
        s = (tau - t0) / max(t1 - t0, eps())
        pos = (1.0 - s) .* p0 .+ s .* p1
        vel = (p1 .- p0) ./ max(t1 - t0, eps())

        ref = zeros(NX_QUAD)
        ref[1:3] .= pos
        ref[4:6] .= vel
        return ref
    end
end

function clearance_planned_reference(track; t_final=10.0,
                                     obstacles=corridor_obstacles(track),
                                     required_clearance=0.0,
                                     nsamples=180,
                                     smoothing_iters=40)
    points = [
        corridor_reference(track, t; t_final=t_final)[1:3]
        for t in range(0.0, t_final; length=nsamples)
    ]

    for _ in 1:smoothing_iters
        new_points = deepcopy(points)

        for i in 2:(nsamples - 1)
            # Smooth first, then enforce clearance. Endpoints remain fixed.
            new_points[i] .= 0.25 .* points[i - 1] .+
                             0.50 .* points[i] .+
                             0.25 .* points[i + 1]

            for obs in obstacles
                inflated_radius = obs.radius + required_clearance
                delta_xy = new_points[i][1:2] .- obs.center[1:2]
                dist_xy = max(norm(delta_xy), 1e-6)
                if dist_xy < inflated_radius
                    push_xy = delta_xy ./ dist_xy .* (inflated_radius - dist_xy + 0.05)
                    new_points[i][1] += push_xy[1]
                    new_points[i][2] += push_xy[2]
                end
            end
        end

        points = new_points
    end

    return _planned_reference_from_points(points, t_final)
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

function _xy_plot_limits(ref, obstacles; pad=0.6)
    xs = collect(ref[1, :])
    ys = collect(ref[2, :])
    for obs in obstacles
        push!(xs, obs.center[1] - obs.radius)
        push!(xs, obs.center[1] + obs.radius)
        push!(ys, obs.center[2] - obs.radius)
        push!(ys, obs.center[2] + obs.radius)
    end
    return (
        (minimum(xs) - pad, maximum(xs) + pad),
        (minimum(ys) - pad, maximum(ys) + pad),
    )
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
    xlim, ylim = _xy_plot_limits(ref, obstacles)

    anim = @animate for idx in idxs
        p = plot(ref[1, :], ref[2, :]; color=:black, ls=:dash, lw=2,
                 label="reference", xlabel="x", ylabel="y", title=plot_title,
                 aspect_ratio=:equal, xlim=xlim, ylim=ylim,
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
                                 obstacles_override=nothing,
                                 obstacle_clearance=0.0,
                                 obstacle_barrier_weight=0.0,
                                 output_tag="$(track)_$(scenario)",
                                 make_gif=true,
                                 quiet=true)
    profiles = scenario_profiles(scenario)
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    obstacles = obstacles_override === nothing ? corridor_obstacles(track) : obstacles_override
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
        obstacles=obstacles,
        obstacle_clearance=obstacle_clearance,
        obstacle_barrier_weight=obstacle_barrier_weight,
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

    title_track = corridor_track_title(track)
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

function run_variable_thrust_corridor_pair(; Ntraj=20,
                                           t_final=10.0,
                                           ilqr_horizon=40,
                                           ilqr_max_iter=2,
                                           max_GPUs=0,
                                           make_gif=true,
                                           quiet=false,
                                           kwargs...)
    return run_corridor_batch(;
        kwargs...,
        tracks=[:s_curve, :sharp_turns],
        scenarios=[:variable_thrust],
        Ntraj=Ntraj,
        t_final=t_final,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        max_GPUs=max_GPUs,
        make_gif=make_gif,
        quiet=quiet,
    )
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
            title_track = corridor_track_title(track)
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

function scale_wind_profile(profile, scale)
    return (
        wind_bias_x = scale * profile.wind_bias_x,
        wind_bias_y = scale * profile.wind_bias_y,
        wind_amp_x = scale * profile.wind_amp_x,
        wind_amp_y = scale * profile.wind_amp_y,
        wind_freq_x_hz = profile.wind_freq_x_hz,
        wind_freq_y_hz = profile.wind_freq_y_hz,
        wind_phase_x = profile.wind_phase_x,
        wind_phase_y = profile.wind_phase_y,
        gust_start_sec = profile.gust_start_sec,
        gust_duration_sec = profile.gust_duration_sec,
        gust_force_x = scale * profile.gust_force_x,
        gust_force_y = scale * profile.gust_force_y,
        gust_force_z = scale * profile.gust_force_z,
    )
end

function wind_profile_for_mph(wind_mph; max_mph=15.0)
    return scale_wind_profile(WIND_GUST_PROFILE, wind_mph / max_mph)
end

function max_tracking_deviation(data, ref_func)
    tvec = data["t"]
    max_dev = 0.0
    for traj in data["u"]
        for (k, state) in enumerate(traj)
            max_dev = max(max_dev, norm(state[1:3] .- ref_func(tvec[k])[1:3]))
        end
    end
    return max_dev
end

function nominal_reference_clearance(ref_func, obstacles, t_final; nsamples=400)
    isempty(obstacles) && return Inf
    return minimum(
        clearance_to_obstacles(ref_func(t)[1:3], obstacles)
        for t in range(0.0, t_final; length=nsamples)
    )
end

function run_wind_envelope_case(; track=:s_curve,
                                wind_mph=15.0,
                                max_mph=15.0,
                                Ntraj=20,
                                t_final=10.0,
                                max_GPUs=0,
                                dt=1e-2,
                                save_stride=5,
                                ilqr_horizon=40,
                                ilqr_max_iter=2,
                                reference_lookahead=0.35,
                                make_gif=false,
                                quiet=false)
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    obstacles = corridor_obstacles(track)
    ref0 = ref_func(0.0)
    wind_profile = wind_profile_for_mph(wind_mph; max_mph=max_mph)
    output_tag = "$(track)_wind_$(round(Int, wind_mph))mph"

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
        wind_profile=wind_profile,
        fault_profile=DEFAULT_FAULT_PROFILE,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        plant_mismatch_enabled=true,
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
    title_track = corridor_track_title(track)
    plot_title = "$title_track: Wind envelope $(wind_mph) mph equivalent"
    analysis = analyze_corridor_run(;
        log_path=log_path,
        ref_func=ref_func,
        obstacles=obstacles,
        output_prefix="quad_corridor_$(output_tag)",
        plot_title=plot_title,
        make_gif=make_gif,
    )

    nom = load(joinpath(log_path, "states_nominal.jld2"))
    tru = load(joinpath(log_path, "states_true.jld2"))
    L1 = load(joinpath(log_path, "states_L1.jld2"))

    return merge(analysis, (
        log_path=log_path,
        wind_mph=wind_mph,
        true_max_deviation=max_tracking_deviation(tru, ref_func),
        L1_max_deviation=max_tracking_deviation(L1, ref_func),
        nominal_max_deviation=max_tracking_deviation(nom, ref_func),
    ))
end

function save_wind_envelope_summary(results; track, vehicle_radius=0.15, margin=0.10)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    wind_speeds = sort(collect(keys(results)))
    true_devs = [results[v].true_max_deviation for v in wind_speeds]
    L1_devs = [results[v].L1_max_deviation for v in wind_speeds]

    worst_true_idx = argmax(true_devs)
    worst_L1_idx = argmax(L1_devs)
    worst_mph = max(wind_speeds[worst_true_idx], wind_speeds[worst_L1_idx])
    worst_deviation = max(maximum(true_devs), maximum(L1_devs))
    required_clearance = worst_deviation + vehicle_radius + margin

    ref_func = t -> corridor_reference(track, t; t_final=results[first(wind_speeds)].metrics.nominal.rmse * 0 + 10.0)
    # Use the saved trajectories' time span for clearance sampling.
    sample_data = load(joinpath(results[first(wind_speeds)].log_path, "states_nominal.jld2"))
    t_final = last(sample_data["t"])
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    nominal_clearance = nominal_reference_clearance(ref_func, corridor_obstacles(track), t_final)

    csv_path = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "quad_corridor_$(track)_wind_envelope_summary.csv")
    open(csv_path, "w") do io
        println(io, "wind_mph,true_max_deviation,L1_max_deviation")
        for (i, wind_mph) in enumerate(wind_speeds)
            println(io, "$(wind_mph),$(true_devs[i]),$(L1_devs[i])")
        end
        println(io)
        println(io, "worst_mph,$worst_mph")
        println(io, "worst_deviation,$worst_deviation")
        println(io, "vehicle_radius,$vehicle_radius")
        println(io, "margin,$margin")
        println(io, "required_clearance,$required_clearance")
        println(io, "nominal_reference_clearance,$nominal_clearance")
        println(io, "clearance_safe,$(nominal_clearance > required_clearance)")
    end

    p = bar(
        string.(wind_speeds),
        [true_devs L1_devs];
        label=["true disturbed iLQR" "L1-DRAC"],
        color=[:firebrick3 :black],
        xlabel="wind speed label (mph equivalent)",
        ylabel="max tracking deviation (m)",
        title="$(track) wind envelope tracking deviation",
        size=(900, 520),
        legend=:topleft,
    )
    hline!(p, [required_clearance]; color=:gray35, lw=2, ls=:dash,
           label="required clearance incl. vehicle+margin")

    plot_path = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "quad_corridor_$(track)_wind_envelope_summary.png")
    savefig(p, plot_path)
    @info "Saved $(basename(csv_path))"
    @info "Saved $(basename(plot_path))"

    return (
        csv=csv_path,
        plot=plot_path,
        worst_mph=worst_mph,
        worst_deviation=worst_deviation,
        required_clearance=required_clearance,
        nominal_reference_clearance=nominal_clearance,
        clearance_safe=nominal_clearance > required_clearance,
    )
end

function make_wind_envelope_gif(results; track, output_prefix="quad_corridor_wind_envelope", nframes=90)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    wind_speeds = sort(collect(keys(results)))
    first_data = load(joinpath(results[first(wind_speeds)].log_path, "states_nominal.jld2"))
    tvec = first_data["t"]
    t_final = last(tvec)
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    ref = _reference_positions(tvec, ref_func)
    idxs = unique(round.(Int, range(1, length(tvec); length=min(nframes, length(tvec)))))

    L1_paths = Dict{Float64, Matrix{Float64}}()
    true_paths = Dict{Float64, Matrix{Float64}}()
    for wind_mph in wind_speeds
        tru = load(joinpath(results[wind_mph].log_path, "states_true.jld2"))
        L1 = load(joinpath(results[wind_mph].log_path, "states_L1.jld2"))
        true_paths[wind_mph] = _mean_positions(tru)
        L1_paths[wind_mph] = _mean_positions(L1)
    end
    xlim, ylim = _xy_plot_limits(ref, corridor_obstacles(track))

    anim = @animate for idx in idxs
        p = plot(ref[1, :], ref[2, :]; color=:black, ls=:dash, lw=2,
                 label="reference", xlabel="x", ylabel="y",
                 title="$(track) wind envelope paths",
                 aspect_ratio=:equal, xlim=xlim, ylim=ylim,
                 size=(760, 560))
        _plot_obstacles_2d!(p, corridor_obstacles(track))

        for wind_mph in wind_speeds
            alpha = 0.25 + 0.65 * wind_mph / maximum(wind_speeds)
            plot!(p, true_paths[wind_mph][1, 1:idx], true_paths[wind_mph][2, 1:idx];
                  color=:firebrick3, alpha=alpha, lw=1.5, label=false)
            plot!(p, L1_paths[wind_mph][1, 1:idx], L1_paths[wind_mph][2, 1:idx];
                  color=:black, alpha=alpha, lw=1.8, label=false)
        end
        plot!(p, [NaN], [NaN]; color=:firebrick3, lw=2, label="true disturbed iLQR")
        plot!(p, [NaN], [NaN]; color=:black, lw=2, label="L1-DRAC")
    end

    filename = "$(output_prefix)_$(track)_wind_sweep.gif"
    gif(anim, joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, filename), fps=15)
    @info "Saved $filename"
    return filename
end

function run_wind_envelope_experiment(; track=:s_curve,
                                      wind_speeds_mph=[5.0, 10.0, 12.0, 15.0],
                                      max_mph=15.0,
                                      Ntraj=20,
                                      t_final=10.0,
                                      max_GPUs=0,
                                      ilqr_horizon=40,
                                      ilqr_max_iter=2,
                                      make_case_gifs=false,
                                      make_sweep_gif=true,
                                      quiet=false)
    results = Dict{Float64, Any}()
    for wind_mph in wind_speeds_mph
        @info "Running wind envelope case" track=track wind_mph=wind_mph
        results[Float64(wind_mph)] = run_wind_envelope_case(;
            track=track,
            wind_mph=Float64(wind_mph),
            max_mph=max_mph,
            Ntraj=Ntraj,
            t_final=t_final,
            max_GPUs=max_GPUs,
            ilqr_horizon=ilqr_horizon,
            ilqr_max_iter=ilqr_max_iter,
            make_gif=make_case_gifs,
            quiet=quiet,
        )
    end

    summary = save_wind_envelope_summary(results; track=track)
    sweep_gif = make_sweep_gif ? make_wind_envelope_gif(results; track=track) : nothing
    return (cases=results, summary=summary, sweep_gif=sweep_gif)
end

function variable_thrust_profile(; label="variable_thrust",
                                 mean_scale=0.50,
                                 amp=0.35,
                                 freq_hz=2.0,
                                 failure_time=2.0,
                                 failed_rotor=1)
    return (
        failure_time = failure_time,
        failed_rotor = failed_rotor,
        failed_rotor_scale = mean_scale,
        sinusoidal_enabled = true,
        sinusoidal_mean_scale = mean_scale,
        sinusoidal_amp = amp,
        sinusoidal_freq_hz = freq_hz,
        label = label,
    )
end

function run_variable_thrust_envelope_case(; track=:s_curve,
                                           profile=variable_thrust_profile(),
                                           Ntraj=20,
                                           t_final=10.0,
                                           max_GPUs=0,
                                           dt=1e-2,
                                           save_stride=5,
                                           ilqr_horizon=40,
                                           ilqr_max_iter=2,
                                           reference_lookahead=0.35,
                                           reference_override=nothing,
                                           obstacle_clearance=0.0,
                                           obstacle_barrier_weight=0.0,
                                           make_gif=false,
                                           quiet=false)
    ref_func = reference_override === nothing ?
        (t -> corridor_reference(track, t; t_final=t_final)) :
        reference_override
    obstacles = corridor_obstacles(track)
    ref0 = ref_func(0.0)
    output_tag = "$(track)_$(profile.label)"

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
        obstacles=obstacles,
        obstacle_clearance=obstacle_clearance,
        obstacle_barrier_weight=obstacle_barrier_weight,
        wind_profile=NO_WIND_PROFILE,
        fault_profile=profile,
        aero_damage_profile=DEFAULT_AERO_DAMAGE_PROFILE,
        plant_mismatch_enabled=true,
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
    title_track = corridor_track_title(track)
    min_scale = profile.sinusoidal_mean_scale - profile.sinusoidal_amp
    max_scale = profile.sinusoidal_mean_scale + profile.sinusoidal_amp
    plot_title = "$title_track: Variable thrust $(profile.label) ($(round(100min_scale; digits=0))%-$(round(100max_scale; digits=0))%)"

    analysis = analyze_corridor_run(;
        log_path=log_path,
        ref_func=ref_func,
        obstacles=obstacles,
        output_prefix="quad_corridor_$(output_tag)",
        plot_title=plot_title,
        make_gif=make_gif,
    )

    nom = load(joinpath(log_path, "states_nominal.jld2"))
    tru = load(joinpath(log_path, "states_true.jld2"))
    L1 = load(joinpath(log_path, "states_L1.jld2"))

    return merge(analysis, (
        log_path=log_path,
        profile=profile,
        true_max_deviation=max_tracking_deviation(tru, ref_func),
        L1_max_deviation=max_tracking_deviation(L1, ref_func),
        nominal_max_deviation=max_tracking_deviation(nom, ref_func),
    ))
end

function default_variable_thrust_envelope_profiles()
    return [
        variable_thrust_profile(; label="vt_mild_40_90_1p25hz",
                                mean_scale=0.65, amp=0.25, freq_hz=1.25),
        variable_thrust_profile(; label="vt_strong_15_85_2hz",
                                mean_scale=0.50, amp=0.35, freq_hz=2.0),
        variable_thrust_profile(; label="vt_severe_10_70_3hz",
                                mean_scale=0.40, amp=0.30, freq_hz=3.0),
    ]
end

function save_variable_thrust_envelope_summary(results; track, vehicle_radius=0.15, margin=0.10)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)
    labels = collect(keys(results))
    true_devs = [results[label].true_max_deviation for label in labels]
    L1_devs = [results[label].L1_max_deviation for label in labels]

    worst_idx = argmax(L1_devs)
    worst_label = labels[worst_idx]
    worst_deviation = L1_devs[worst_idx]
    required_clearance = worst_deviation + vehicle_radius + margin

    sample_data = load(joinpath(results[worst_label].log_path, "states_nominal.jld2"))
    t_final = last(sample_data["t"])
    ref_func = t -> corridor_reference(track, t; t_final=t_final)
    nominal_clearance = nominal_reference_clearance(ref_func, corridor_obstacles(track), t_final)

    csv_path = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "quad_corridor_$(track)_variable_thrust_envelope_summary.csv")
    open(csv_path, "w") do io
        println(io, "case,true_max_deviation,L1_max_deviation,min_effectiveness,max_effectiveness,freq_hz")
        for label in labels
            profile = results[label].profile
            min_scale = profile.sinusoidal_mean_scale - profile.sinusoidal_amp
            max_scale = profile.sinusoidal_mean_scale + profile.sinusoidal_amp
            println(io, "$(label),$(results[label].true_max_deviation),$(results[label].L1_max_deviation),$(min_scale),$(max_scale),$(profile.sinusoidal_freq_hz)")
        end
        println(io)
        println(io, "worst_case_for_L1_planning,$worst_label")
        println(io, "L1_worst_deviation,$worst_deviation")
        println(io, "vehicle_radius,$vehicle_radius")
        println(io, "margin,$margin")
        println(io, "L1_required_clearance,$required_clearance")
        println(io, "nominal_reference_clearance,$nominal_clearance")
        println(io, "clearance_safe,$(nominal_clearance > required_clearance)")
    end

    p = bar(
        labels,
        [true_devs L1_devs];
        label=["true disturbed iLQR" "L1-DRAC"],
        color=[:firebrick3 :black],
        xlabel="variable thrust case",
        ylabel="max tracking deviation (m)",
        title="$(track) variable thrust envelope",
        xrotation=20,
        size=(1100, 560),
        legend=:topleft,
    )
    hline!(p, [required_clearance]; color=:gray35, lw=2, ls=:dash,
           label="L1 required clearance incl. vehicle+margin")

    plot_path = joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, "quad_corridor_$(track)_variable_thrust_envelope_summary.png")
    savefig(p, plot_path)
    @info "Saved $(basename(csv_path))"
    @info "Saved $(basename(plot_path))"

    return (
        csv=csv_path,
        plot=plot_path,
        worst_case=worst_label,
        worst_profile=results[worst_label].profile,
        worst_deviation=worst_deviation,
        required_clearance=required_clearance,
        nominal_reference_clearance=nominal_clearance,
        clearance_safe=nominal_clearance > required_clearance,
    )
end

function run_variable_thrust_envelope_experiment(; track=:s_curve,
                                                 profiles=default_variable_thrust_envelope_profiles(),
                                                 Ntraj=20,
                                                 t_final=10.0,
                                                 max_GPUs=0,
                                                 ilqr_horizon=40,
                                                 ilqr_max_iter=2,
                                                 make_case_gifs=false,
                                                 quiet=false)
    results = Dict{String, Any}()
    for profile in profiles
        @info "Running variable thrust envelope case" track=track case=profile.label
        results[string(profile.label)] = run_variable_thrust_envelope_case(;
            track=track,
            profile=profile,
            Ntraj=Ntraj,
            t_final=t_final,
            max_GPUs=max_GPUs,
            ilqr_horizon=ilqr_horizon,
            ilqr_max_iter=ilqr_max_iter,
            make_gif=make_case_gifs,
            quiet=quiet,
        )
    end

    summary = save_variable_thrust_envelope_summary(results; track=track)
    return (cases=results, summary=summary)
end

function run_variable_thrust_clearance_planned_case(; track=:s_curve,
                                                    envelope_result,
                                                    Ntraj=20,
                                                    t_final=10.0,
                                                    max_GPUs=0,
                                                    ilqr_horizon=40,
                                                    ilqr_max_iter=4,
                                                    obstacle_barrier_weight=80.0,
                                                    make_gif=true,
                                                    quiet=false)
    profile = envelope_result.summary.worst_profile
    required_clearance = envelope_result.summary.required_clearance
    @info "Running clearance-planned variable thrust case" track=track case=profile.label required_clearance=required_clearance
    planned_ref = clearance_planned_reference(
        track;
        t_final=t_final,
        obstacles=corridor_obstacles(track),
        required_clearance=required_clearance,
    )

    return run_variable_thrust_envelope_case(;
        track=track,
        profile=merge(profile, (label="$(profile.label)_clearance_planned",)),
        Ntraj=Ntraj,
        t_final=t_final,
        max_GPUs=max_GPUs,
        ilqr_horizon=ilqr_horizon,
        ilqr_max_iter=ilqr_max_iter,
        reference_override=planned_ref,
        obstacle_clearance=required_clearance,
        obstacle_barrier_weight=obstacle_barrier_weight,
        make_gif=make_gif,
        quiet=quiet,
    )
end

function make_variable_thrust_condition_gif(results; track,
                                            controller=:L1_DRAC,
                                            output_prefix="quad_corridor_variable_thrust_planned",
                                            reference_override=nothing,
                                            nframes=90)
    mkpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR)

    labels = collect(keys(results))
    first_data = load(joinpath(results[first(labels)].log_path, "states_nominal.jld2"))
    tvec = first_data["t"]
    t_final = last(tvec)
    ref_func = reference_override === nothing ?
        (t -> corridor_reference(track, t; t_final=t_final)) :
        reference_override
    ref = _reference_positions(tvec, ref_func)
    idxs = unique(round.(Int, range(1, length(tvec); length=min(nframes, length(tvec)))))

    state_file = controller == :L1_DRAC ? "states_L1.jld2" : "states_true.jld2"
    controller_label = controller == :L1_DRAC ? "L1-DRAC" : "true disturbed iLQR"
    controller_color = controller == :L1_DRAC ? :black : :firebrick3

    paths = Dict{String, Matrix{Float64}}()
    for label in labels
        data = load(joinpath(results[label].log_path, state_file))
        paths[label] = _mean_positions(data)
    end
    xlim, ylim = _xy_plot_limits(ref, corridor_obstacles(track))

    anim = @animate for idx in idxs
        p = plot(ref[1, :], ref[2, :]; color=:black, ls=:dash, lw=2,
                 label="planned reference", xlabel="x", ylabel="y",
                 title="$(controller_label): variable thrust fault sweep",
                 aspect_ratio=:equal, xlim=xlim, ylim=ylim,
                 size=(760, 560))
        _plot_obstacles_2d!(p, corridor_obstacles(track))

        for (i, label) in enumerate(labels)
            alpha = 0.35 + 0.55 * i / max(length(labels), 1)
            plot!(p, paths[label][1, 1:idx], paths[label][2, 1:idx];
                  color=controller_color, alpha=alpha, lw=1.8, label=false)
            scatter!(p, [paths[label][1, idx]], [paths[label][2, idx]];
                     color=controller_color, alpha=alpha, ms=3, label=false)
        end

        plot!(p, [NaN], [NaN]; color=controller_color, lw=2.5, label=controller_label)
    end

    suffix = controller == :L1_DRAC ? "L1" : "true_disturbed"
    filename = "$(output_prefix)_$(track)_$(suffix)_motor_fault_sweep.gif"
    gif(anim, joinpath(OBSTACLE_AVOIDANCE_OUTPUT_DIR, filename), fps=15)
    @info "Saved $filename"
    return filename
end

function run_variable_thrust_clearance_planned_sweep(; track=:s_curve,
                                                     envelope_result,
                                                     profiles=default_variable_thrust_envelope_profiles(),
                                                     Ntraj=20,
                                                     t_final=10.0,
                                                     max_GPUs=0,
                                                     ilqr_horizon=40,
                                                     ilqr_max_iter=4,
                                                     obstacle_barrier_weight=80.0,
                                                     make_gifs=true,
                                                     quiet=false)
    required_clearance = envelope_result.summary.required_clearance
    planned_ref = clearance_planned_reference(
        track;
        t_final=t_final,
        obstacles=corridor_obstacles(track),
        required_clearance=required_clearance,
    )
    results = Dict{String, Any}()

    for profile in profiles
        planned_profile = merge(profile, (label="$(profile.label)_clearance_planned",))
        @info "Running clearance-planned motor fault sweep case" track=track case=planned_profile.label required_clearance=required_clearance
        results[string(profile.label)] = run_variable_thrust_envelope_case(;
            track=track,
            profile=planned_profile,
            Ntraj=Ntraj,
            t_final=t_final,
            max_GPUs=max_GPUs,
            ilqr_horizon=ilqr_horizon,
            ilqr_max_iter=ilqr_max_iter,
            reference_override=planned_ref,
            obstacle_clearance=required_clearance,
            obstacle_barrier_weight=obstacle_barrier_weight,
            make_gif=false,
            quiet=quiet,
        )
    end

    gif_L1 = make_gifs ? make_variable_thrust_condition_gif(results; track=track, controller=:L1_DRAC,
                                                             reference_override=planned_ref) : nothing
    gif_true = make_gifs ? make_variable_thrust_condition_gif(results; track=track, controller=:true_disturbed,
                                                               reference_override=planned_ref) : nothing

    return (
        cases=results,
        required_clearance=required_clearance,
        L1_gif=gif_L1,
        true_disturbed_gif=gif_true,
    )
end
