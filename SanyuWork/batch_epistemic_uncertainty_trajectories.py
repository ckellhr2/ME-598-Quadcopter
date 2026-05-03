import os
import argparse

import numpy as np
import pandas as pd
import pybullet as p

from NonIdealSystem import run_simulation


GOAL_POS = [1.3, 2.1, 1.4]
NOMINAL_VALUES = {
    "mass": 1.0,
    "inertia": 1.0,
    "thrust": 1.0,
    "yaw_coeff": 1.0,
}

# Fixed plant mismatch for each rollout. Tuned to show visible separation
# while still being reasonably recoverable with the nominal controller.
UNCERTAINTY_BOUNDS = {
    "mass": (1.20, 1.35),
    "inertia": (1.15, 1.30),
    "thrust": (0.80, 0.92),
    "yaw_coeff": (0.80, 1.20),
}

SCENARIOS = [
    {"name": "mass_only", "mass": True, "inertia": False, "thrust": False, "yaw_coeff": False},
    {"name": "thrust_only", "mass": False, "inertia": False, "thrust": True, "yaw_coeff": False},
    {"name": "inertia_only", "mass": False, "inertia": True, "thrust": False, "yaw_coeff": False},
    {"name": "yaw_coeff_only", "mass": False, "inertia": False, "thrust": False, "yaw_coeff": True},
    {"name": "mass_thrust_combo", "mass": True, "inertia": False, "thrust": True, "yaw_coeff": False},
]


def sample_all_uncertainties(rng):
    return {
        "mass": float(rng.uniform(*UNCERTAINTY_BOUNDS["mass"])),
        "inertia": float(rng.uniform(*UNCERTAINTY_BOUNDS["inertia"])),
        "thrust": float(rng.uniform(*UNCERTAINTY_BOUNDS["thrust"])),
        "yaw_coeff": float(rng.uniform(*UNCERTAINTY_BOUNDS["yaw_coeff"])),
    }


def apply_scenario_uncertainties(all_uncertainties, scenario):
    return {
        "mass": all_uncertainties["mass"] if scenario["mass"] else NOMINAL_VALUES["mass"],
        "inertia": all_uncertainties["inertia"] if scenario["inertia"] else NOMINAL_VALUES["inertia"],
        "thrust": all_uncertainties["thrust"] if scenario["thrust"] else NOMINAL_VALUES["thrust"],
        "yaw_coeff": all_uncertainties["yaw_coeff"] if scenario["yaw_coeff"] else NOMINAL_VALUES["yaw_coeff"],
    }


def write_outputs(trajectory_rows, output_csv_path, output_pickle_path, output_summary_path,
                  scenarios_to_run, nominal_initial_conditions, nominal_csv_path):
    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(output_csv_path, index=False)
    trajectory_df.to_pickle(output_pickle_path)

    summary_lines = [
        "Fixed epistemic uncertainty batch checkpoint.",
        f"Scenarios requested: {len(scenarios_to_run)}",
        f"Trials requested per scenario: {len(nominal_initial_conditions)}",
        f"Total stored rows: {len(trajectory_df)}",
        f"Completed scenarios in file: {trajectory_df['scenario'].nunique() if not trajectory_df.empty else 0}",
        f"Completed trial groups in file: {trajectory_df[['scenario', 'trial']].drop_duplicates().shape[0] if not trajectory_df.empty else 0}",
        f"Goal position: {GOAL_POS}",
        f"Nominal values: {NOMINAL_VALUES}",
        f"Fixed uncertainty bounds: {UNCERTAINTY_BOUNDS}",
        "Scenarios requested:",
    ]
    for scenario in scenarios_to_run:
        summary_lines.append(f"  - {scenario['name']}")
    if not trajectory_df.empty:
        summary_lines.append(f"Max time step saved: {trajectory_df['time_step'].max()}")
    summary_lines.extend([
        f"Initial conditions reused from: {nominal_csv_path}",
        f"CSV saved to: {output_csv_path}",
        f"Pickle saved to: {output_pickle_path}",
    ])

    with open(output_summary_path, "w", encoding="ascii") as f:
        f.write("\n".join(summary_lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run fixed epistemic uncertainty trajectory batches")
    parser.add_argument("--scenario", default=None, help="Optional scenario name to run")
    parser.add_argument("--max-trials", type=int, default=50, help="Optional cap on reused nominal trials")
    parser.add_argument("--output-csv", default="epistemic_uncertainty_trajectories.csv", help="Output CSV path")
    parser.add_argument("--output-pkl", default="epistemic_uncertainty_trajectories.pkl", help="Output pickle path")
    parser.add_argument("--output-summary", default="epistemic_uncertainty_trajectories_summary.txt", help="Output summary path")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="Write partial outputs every N completed runs")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    base_seed = 598
    rng = np.random.default_rng(base_seed)
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
        .reset_index(drop=True)
    )
    if args.max_trials is not None and args.max_trials > 0:
        nominal_initial_conditions = nominal_initial_conditions.head(args.max_trials)

    scenarios_to_run = SCENARIOS
    if args.scenario:
        scenarios_to_run = [scenario for scenario in SCENARIOS if scenario["name"] == args.scenario]
        if not scenarios_to_run:
            available = ", ".join(scenario["name"] for scenario in SCENARIOS)
            raise ValueError(f"Unknown scenario '{args.scenario}'. Available scenarios: {available}")

    physics_client = p.connect(p.DIRECT)
    trajectory_rows = []
    total_runs = len(nominal_initial_conditions) * len(scenarios_to_run)

    try:
        print("Starting fixed epistemic uncertainty batch.")
        print(f"Initial conditions reused from: {nominal_csv_path}")
        print(f"Trials per scenario: {len(nominal_initial_conditions)}")
        print(f"Goal position: {GOAL_POS}")
        print(f"Fixed uncertainty bounds: {UNCERTAINTY_BOUNDS}")
        print("Scenarios:")
        for scenario in scenarios_to_run:
            print(f"  - {scenario['name']}")

        completed_runs = 0
        for scenario in scenarios_to_run:
            for _, row in nominal_initial_conditions.iterrows():
                trial_index = int(row["trial"])
                seed = int(row["seed"])
                start_pos = [float(row["start_x"]), float(row["start_y"]), float(row["start_z"])]

                all_uncertainties = sample_all_uncertainties(rng)
                uncertainty = apply_scenario_uncertainties(all_uncertainties, scenario)

                result = run_simulation(
                    start_pos=start_pos,
                    x_goal_pos=GOAL_POS,
                    connection_mode=p.DIRECT,
                    keep_alive=False,
                    max_steps=12 * 240,
                    log_interval=0,
                    verbose=False,
                    reuse_existing_connection=True,
                    plant_mass_scale=uncertainty["mass"],
                    plant_inertia_scale=uncertainty["inertia"],
                    plant_thrust_scale=uncertainty["thrust"],
                    plant_yaw_coeff_scale=uncertainty["yaw_coeff"],
                )

                for row_data in result["trajectory_rows"]:
                    trajectory_rows.append({
                        "scenario": scenario["name"],
                        "trial": trial_index,
                        "seed": seed,
                        "start_x": float(start_pos[0]),
                        "start_y": float(start_pos[1]),
                        "start_z": float(start_pos[2]),
                        "goal_x": float(GOAL_POS[0]),
                        "goal_y": float(GOAL_POS[1]),
                        "goal_z": float(GOAL_POS[2]),
                        "unc_mass": float(uncertainty["mass"]),
                        "unc_inertia": float(uncertainty["inertia"]),
                        "unc_thrust": float(uncertainty["thrust"]),
                        "unc_yaw_coeff": float(uncertainty["yaw_coeff"]),
                        "time_step": int(row_data["time_step"]),
                        "time_sec": float(row_data["time_sec"]),
                        "x": float(row_data["x"]),
                        "y": float(row_data["y"]),
                        "z": float(row_data["z"]),
                        "effective_mass_scale": float(row_data["effective_mass_scale"]),
                        "effective_inertia_scale": float(row_data["effective_inertia_scale"]),
                        "effective_thrust_scale": float(row_data["effective_thrust_scale"]),
                        "effective_yaw_coeff_scale": float(row_data["effective_yaw_coeff_scale"]),
                    })

                completed_runs += 1
                print(
                    f"Completed run {completed_runs}/{total_runs}... "
                    f"scenario={scenario['name']} trial={trial_index} "
                    f"steps={result['steps']} reached_goal={result['reached_goal']}",
                    flush=True,
                )

                if args.checkpoint_every > 0 and completed_runs % args.checkpoint_every == 0:
                    write_outputs(
                        trajectory_rows,
                        output_csv_path,
                        output_pickle_path,
                        output_summary_path,
                        scenarios_to_run,
                        nominal_initial_conditions,
                        nominal_csv_path,
                    )

    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)

    write_outputs(
        trajectory_rows,
        output_csv_path,
        output_pickle_path,
        output_summary_path,
        scenarios_to_run,
        nominal_initial_conditions,
        nominal_csv_path,
    )

    trajectory_df = pd.DataFrame(trajectory_rows)
    print("\nFixed epistemic uncertainty batch complete.")
    print(f"Scenarios: {trajectory_df['scenario'].nunique() if not trajectory_df.empty else 0}")
    print(f"Trials per scenario: {trajectory_df['trial'].nunique() if not trajectory_df.empty else 0}")
    print(f"Stored rows: {len(trajectory_df)}")
    print(f"CSV saved to: {output_csv_path}")
    print(f"Pickle saved to: {output_pickle_path}")
    print(f"Summary saved to: {output_summary_path}")


if __name__ == "__main__":
    main()
