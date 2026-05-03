import os
import argparse

import numpy as np
import pandas as pd
import pybullet as p

from TimeVaryingUncertainSystem import run_simulation


GOAL_POS = [1.3, 2.1, 1.4]
SCENARIOS = [
    "wind_only",
    "thrust_drop_only",
]


def sample_disturbance_profile(rng, scenario):
    profile = {
        "thrust_drop_fraction": float(rng.uniform(0.02, 0.06)),
        "thrust_drop_start_sec": float(rng.uniform(1.2, 2.2)),
        "thrust_drop_end_sec": float(rng.uniform(2.6, 3.8)),
        "wind_bias_x": float(rng.uniform(-0.04, 0.04)),
        "wind_bias_y": float(rng.uniform(-0.04, 0.04)),
        "wind_amp_x": float(rng.uniform(0.015, 0.05)),
        "wind_amp_y": float(rng.uniform(0.015, 0.05)),
        "wind_freq_x_hz": float(rng.uniform(0.10, 0.22)),
        "wind_freq_y_hz": float(rng.uniform(0.10, 0.22)),
        "wind_phase_x": float(rng.uniform(0.0, 2.0 * np.pi)),
        "wind_phase_y": float(rng.uniform(0.0, 2.0 * np.pi)),
        "gust_start_sec": float(rng.uniform(1.4, 2.6)),
        "gust_duration_sec": float(rng.uniform(0.25, 0.55)),
        "gust_force_x": float(rng.uniform(-0.08, 0.08)),
        "gust_force_y": float(rng.uniform(-0.08, 0.08)),
        "gust_force_z": float(rng.uniform(-0.03, 0.03)),
    }

    if scenario == "wind_only":
        profile["thrust_drop_fraction"] = 0.0
        profile["gust_force_x"] = 0.0
        profile["gust_force_y"] = 0.0
        profile["gust_force_z"] = 0.0
    elif scenario == "thrust_drop_only":
        profile["wind_bias_x"] = 0.0
        profile["wind_bias_y"] = 0.0
        profile["wind_amp_x"] = 0.0
        profile["wind_amp_y"] = 0.0
        profile["gust_force_x"] = 0.0
        profile["gust_force_y"] = 0.0
        profile["gust_force_z"] = 0.0
    else:
        available = ", ".join(SCENARIOS)
        raise ValueError(f"Unknown scenario '{scenario}'. Available scenarios: {available}")

    return profile


def write_outputs(trajectory_rows, output_csv_path, output_pickle_path, output_summary_path, trials_requested):
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(output_csv_path, index=False)
    trajectory_df.to_pickle(output_pickle_path)

    summary_lines = [
        "Time-varying uncertainty batch checkpoint.",
        f"Trials requested: {trials_requested}",
        f"Stored rows: {len(trajectory_df)}",
        f"Completed trials: {trajectory_df['trial'].nunique() if not trajectory_df.empty else 0}",
        f"Goal position: {GOAL_POS}",
    ]
    if not trajectory_df.empty:
        summary_lines.extend([
            f"Max time step saved: {trajectory_df['time_step'].max()}",
            f"Scenarios present: {trajectory_df['scenario'].nunique()}",
        ])
    summary_lines.extend([
        f"CSV saved to: {output_csv_path}",
        f"Pickle saved to: {output_pickle_path}",
    ])

    with open(output_summary_path, "w", encoding="ascii") as f:
        f.write("\n".join(summary_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run time-varying plant uncertainty trajectories")
    parser.add_argument("--max-trials", type=int, default=50, help="Number of reused nominal initial conditions to run")
    parser.add_argument("--scenario", default=None, help="Optional scenario name to run")
    parser.add_argument("--output-csv", default="time_varying_uncertainty_trajectories.csv", help="Output CSV path")
    parser.add_argument("--output-pkl", default="time_varying_uncertainty_trajectories.pkl", help="Output pickle path")
    parser.add_argument("--output-summary", default="time_varying_uncertainty_trajectories_summary.txt", help="Output summary path")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial outputs every N completed trials")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    base_seed = 598
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nominal_csv_path = os.path.join(script_dir, "nominal_trajectory_positions.csv")
    output_csv_path = os.path.abspath(args.output_csv)
    output_pickle_path = os.path.abspath(args.output_pkl)
    output_summary_path = os.path.abspath(args.output_summary)

    if not os.path.exists(nominal_csv_path):
        raise FileNotFoundError(
            f"Nominal trajectory CSV not found: {nominal_csv_path}\n"
            "Run batch_nominal_trajectory_positions.py first."
        )

    nominal_df = pd.read_csv(nominal_csv_path)
    nominal_initial_conditions = (
        nominal_df[["trial", "seed", "start_x", "start_y", "start_z"]]
        .drop_duplicates(subset=["trial"])
        .sort_values("trial")
        .head(args.max_trials)
        .reset_index(drop=True)
    )
    scenarios_to_run = SCENARIOS
    if args.scenario:
        scenarios_to_run = [scenario for scenario in SCENARIOS if scenario == args.scenario]
        if not scenarios_to_run:
            available = ", ".join(SCENARIOS)
            raise ValueError(f"Unknown scenario '{args.scenario}'. Available scenarios: {available}")

    physics_client = p.connect(p.DIRECT)
    trajectory_rows = []
    total_runs = len(nominal_initial_conditions) * len(scenarios_to_run)

    try:
        print("Starting time-varying uncertainty batch.")
        print(f"Initial conditions reused from: {nominal_csv_path}")
        print(f"Trials to run: {len(nominal_initial_conditions)}")
        print(f"Goal position: {GOAL_POS}")
        print("Scenarios:")
        for scenario in scenarios_to_run:
            print(f"  - {scenario}")

        completed_runs = 0
        for scenario in scenarios_to_run:
            for run_index, row in nominal_initial_conditions.iterrows():
                trial_index = int(row["trial"])
                seed = int(row["seed"])
                start_pos = [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])]
                rng = np.random.default_rng(base_seed + seed)
                profile = sample_disturbance_profile(rng, scenario)

                result = run_simulation(
                    start_pos=start_pos,
                    x_goal_pos=GOAL_POS,
                    disturbance_profile=profile,
                    connection_mode=p.DIRECT,
                    keep_alive=False,
                    max_steps=12 * 240,
                    log_interval=0,
                    verbose=False,
                    reuse_existing_connection=True,
                )

                for row_data in result["trajectory_rows"]:
                    trajectory_rows.append({
                        "scenario": scenario,
                        "trial": trial_index,
                        "seed": seed,
                        "start_x": float(start_pos[0]),
                        "start_y": float(start_pos[1]),
                        "start_z": float(start_pos[2]),
                        "goal_x": float(GOAL_POS[0]),
                        "goal_y": float(GOAL_POS[1]),
                        "goal_z": float(GOAL_POS[2]),
                        "thrust_drop_fraction": float(profile["thrust_drop_fraction"]),
                        "thrust_drop_start_sec": float(profile["thrust_drop_start_sec"]),
                        "thrust_drop_end_sec": float(profile["thrust_drop_end_sec"]),
                        "wind_bias_x": float(profile["wind_bias_x"]),
                        "wind_bias_y": float(profile["wind_bias_y"]),
                        "wind_amp_x": float(profile["wind_amp_x"]),
                        "wind_amp_y": float(profile["wind_amp_y"]),
                        "wind_freq_x_hz": float(profile["wind_freq_x_hz"]),
                        "wind_freq_y_hz": float(profile["wind_freq_y_hz"]),
                        "wind_phase_x": float(profile["wind_phase_x"]),
                        "wind_phase_y": float(profile["wind_phase_y"]),
                        "gust_start_sec": float(profile["gust_start_sec"]),
                        "gust_duration_sec": float(profile["gust_duration_sec"]),
                        "gust_force_x": float(profile["gust_force_x"]),
                        "gust_force_y": float(profile["gust_force_y"]),
                        "gust_force_z": float(profile["gust_force_z"]),
                        "time_step": int(row_data["time_step"]),
                        "time_sec": float(row_data["time_sec"]),
                        "x": float(row_data["x"]),
                        "y": float(row_data["y"]),
                        "z": float(row_data["z"]),
                        "effective_thrust_scale": float(row_data["effective_thrust_scale"]),
                        "wind_fx": float(row_data["wind_fx"]),
                        "wind_fy": float(row_data["wind_fy"]),
                        "wind_fz": float(row_data["wind_fz"]),
                    })

                completed_runs += 1
                print(
                    f"Completed run {completed_runs}/{total_runs}... "
                    f"scenario={scenario} trial={trial_index} "
                    f"steps={result['steps']} reached_goal={result['reached_goal']}",
                    flush=True,
                )

                if args.checkpoint_every > 0 and completed_runs % args.checkpoint_every == 0:
                    write_outputs(
                        trajectory_rows,
                        output_csv_path,
                        output_pickle_path,
                        output_summary_path,
                        total_runs,
                    )

    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)

    write_outputs(
        trajectory_rows,
        output_csv_path,
        output_pickle_path,
        output_summary_path,
        total_runs,
    )

    trajectory_df = pd.DataFrame(trajectory_rows)
    print("\nTime-varying uncertainty batch complete.")
    print(f"Trials: {trajectory_df['trial'].nunique() if not trajectory_df.empty else 0}")
    print(f"Stored rows: {len(trajectory_df)}")
    print(f"CSV saved to: {output_csv_path}")
    print(f"Pickle saved to: {output_pickle_path}")
    print(f"Summary saved to: {output_summary_path}")


if __name__ == "__main__":
    main()
