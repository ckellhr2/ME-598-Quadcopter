import os
import argparse

import numpy as np
import pandas as pd


def empirical_wasserstein_1d(samples_a, samples_b):
    """Compute empirical 1-Wasserstein distance between two 1D sample sets."""
    a = np.sort(np.asarray(samples_a, dtype=float))
    b = np.sort(np.asarray(samples_b, dtype=float))

    if a.size == 0 or b.size == 0:
        return np.nan

    values = np.concatenate([a, b])
    values.sort()
    if values.size <= 1:
        return 0.0

    deltas = np.diff(values)
    if deltas.size == 0:
        return 0.0

    cdf_a = np.searchsorted(a, values[:-1], side="right") / a.size
    cdf_b = np.searchsorted(b, values[:-1], side="right") / b.size
    return float(np.sum(np.abs(cdf_a - cdf_b) * deltas))


<<<<<<< HEAD
=======
<<<<<<< HEAD
def compute_wasserstein_rows(uncertain_df, nominal_df, scenario_filter=None):
    required_columns = {"trial", "time_step", "time_sec", "x", "y", "z"}
    if "scenario" not in uncertain_df.columns:
        raise ValueError("Uncertainty CSV missing required column: scenario")
    if not required_columns.issubset(uncertain_df.columns):
        raise ValueError("Uncertainty CSV missing one or more required columns")
    if not required_columns.issubset(nominal_df.columns):
        raise ValueError("Nominal CSV missing one or more required columns")

    scenarios = sorted(uncertain_df["scenario"].unique())
    if scenario_filter:
        scenarios = [scenario for scenario in scenarios if scenario == scenario_filter]
        if not scenarios:
            raise ValueError(f"Scenario '{scenario_filter}' not found in uncertainty CSV")

    results = []
    nominal_time_steps = set(nominal_df["time_step"].unique())

    for scenario in scenarios:
        scenario_df = uncertain_df[uncertain_df["scenario"] == scenario]
=======
>>>>>>> 2a05eebd2a3f3593b48c6bdea5f7e21e5df1c138
def main():
    parser = argparse.ArgumentParser(
        description="Compute per-time-step empirical Wasserstein-1 distances between nominal and uncertain trajectories"
    )
    parser.add_argument(
        "--epistemic-csv",
        default="epistemic_uncertainty_trajectories.csv",
        help="Path to the epistemic uncertainty trajectory CSV file",
    )
    parser.add_argument(
        "--nominal-csv",
        default="nominal_trajectory_positions.csv",
        help="Path to the nominal trajectory CSV file",
    )
    parser.add_argument(
        "--output-csv",
        default="wasserstein_time_series.csv",
        help="Output CSV path for the Wasserstein time series",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    epistemic_csv_path = os.path.abspath(args.epistemic_csv)
    nominal_csv_path = os.path.abspath(args.nominal_csv)
    output_csv_path = os.path.abspath(args.output_csv)

    if not os.path.exists(epistemic_csv_path):
        raise FileNotFoundError(f"Epistemic CSV file not found: {epistemic_csv_path}")
    if not os.path.exists(nominal_csv_path):
        raise FileNotFoundError(f"Nominal CSV file not found: {nominal_csv_path}")

    epistemic_df = pd.read_csv(epistemic_csv_path)
    nominal_df = pd.read_csv(nominal_csv_path)

    required_columns = {"trial", "time_step", "time_sec", "x", "y", "z"}
    if "scenario" not in epistemic_df.columns:
        raise ValueError("Epistemic CSV missing required column: scenario")
    if not required_columns.issubset(epistemic_df.columns):
        raise ValueError("Epistemic CSV missing one or more required columns")
    if not required_columns.issubset(nominal_df.columns):
        raise ValueError("Nominal CSV missing one or more required columns")

    scenarios = sorted(epistemic_df["scenario"].unique())
    results = []

    nominal_time_steps = set(nominal_df["time_step"].unique())

    for scenario in scenarios:
        scenario_df = epistemic_df[epistemic_df["scenario"] == scenario]
<<<<<<< HEAD
=======
>>>>>>> 8db271904d96e850fb7a44f8d5bd61e315fd150b
>>>>>>> 2a05eebd2a3f3593b48c6bdea5f7e21e5df1c138
        common_time_steps = sorted(nominal_time_steps & set(scenario_df["time_step"].unique()))

        for time_step in common_time_steps:
            nominal_t = nominal_df[nominal_df["time_step"] == time_step]
            uncertain_t = scenario_df[scenario_df["time_step"] == time_step]

            if nominal_t.empty or uncertain_t.empty:
                continue

            row = {
                "scenario": scenario,
                "time_step": int(time_step),
                "time_sec": float(uncertain_t["time_sec"].iloc[0]),
                "num_nominal_samples": int(len(nominal_t)),
                "num_uncertain_samples": int(len(uncertain_t)),
                "wasserstein_x": empirical_wasserstein_1d(nominal_t["x"], uncertain_t["x"]),
                "wasserstein_y": empirical_wasserstein_1d(nominal_t["y"], uncertain_t["y"]),
                "wasserstein_z": empirical_wasserstein_1d(nominal_t["z"], uncertain_t["z"]),
            }
            row["wasserstein_xyz_mean"] = float(
                np.nanmean([row["wasserstein_x"], row["wasserstein_y"], row["wasserstein_z"]])
            )
            results.append(row)

<<<<<<< HEAD
    results_df = pd.DataFrame(results).sort_values(["scenario", "time_step"]).reset_index(drop=True)
    results_df.to_csv(output_csv_path, index=False)

=======
<<<<<<< HEAD
    return pd.DataFrame(results).sort_values(["scenario", "time_step"]).reset_index(drop=True)


def save_wasserstein_csv(uncertain_csv_path, nominal_csv_path, output_csv_path, scenario_filter=None):
    uncertain_df = pd.read_csv(uncertain_csv_path)
    nominal_df = pd.read_csv(nominal_csv_path)
    results_df = compute_wasserstein_rows(uncertain_df, nominal_df, scenario_filter=scenario_filter)
    results_df.to_csv(output_csv_path, index=False)
=======
    results_df = pd.DataFrame(results).sort_values(["scenario", "time_step"]).reset_index(drop=True)
    results_df.to_csv(output_csv_path, index=False)

>>>>>>> 8db271904d96e850fb7a44f8d5bd61e315fd150b
>>>>>>> 2a05eebd2a3f3593b48c6bdea5f7e21e5df1c138
    print(f"Saved Wasserstein time series to: {output_csv_path}")
    print(f"Scenarios: {results_df['scenario'].nunique() if not results_df.empty else 0}")
    print(f"Rows: {len(results_df)}")


<<<<<<< HEAD
=======
<<<<<<< HEAD
def main():
    parser = argparse.ArgumentParser(
        description="Compute per-time-step empirical Wasserstein-1 distances between nominal and uncertain trajectories"
    )
    parser.add_argument(
        "--uncertain-csv",
        default="epistemic_uncertainty_trajectories.csv",
        help="Path to the uncertainty trajectory CSV file",
    )
    parser.add_argument(
        "--nominal-csv",
        default="nominal_trajectory_positions.csv",
        help="Path to the nominal trajectory CSV file",
    )
    parser.add_argument(
        "--output-csv",
        default="wasserstein_time_series.csv",
        help="Output CSV path for the Wasserstein time series",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name to compute",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    nominal_csv_path = os.path.abspath(args.nominal_csv)
    if not os.path.exists(nominal_csv_path):
        raise FileNotFoundError(f"Nominal CSV file not found: {nominal_csv_path}")

    explicit_uncertain_csv = args.uncertain_csv != "epistemic_uncertainty_trajectories.csv" or args.output_csv != "wasserstein_time_series.csv" or args.scenario is not None
    if explicit_uncertain_csv:
        uncertain_csv_path = os.path.abspath(args.uncertain_csv)
        output_csv_path = os.path.abspath(args.output_csv)
        if not os.path.exists(uncertain_csv_path):
            raise FileNotFoundError(f"Uncertainty CSV file not found: {uncertain_csv_path}")
        save_wasserstein_csv(
            uncertain_csv_path,
            nominal_csv_path,
            output_csv_path,
            scenario_filter=args.scenario,
        )
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_jobs = [
        ("epistemic_uncertainty_trajectories.csv", "wasserstein_epistemic_time_series.csv"),
        ("time_varying_uncertainty_trajectories.csv", "wasserstein_time_varying_time_series.csv"),
    ]

    ran_any = False
    for uncertain_name, output_name in default_jobs:
        uncertain_csv_path = os.path.join(script_dir, uncertain_name)
        output_csv_path = os.path.join(script_dir, output_name)
        if os.path.exists(uncertain_csv_path):
            save_wasserstein_csv(
                uncertain_csv_path,
                nominal_csv_path,
                output_csv_path,
                scenario_filter=None,
            )
            ran_any = True

    if not ran_any:
        raise FileNotFoundError(
            "No default uncertainty CSVs were found. Expected one of: "
            "epistemic_uncertainty_trajectories.csv, time_varying_uncertainty_trajectories.csv"
        )


=======
>>>>>>> 8db271904d96e850fb7a44f8d5bd61e315fd150b
>>>>>>> 2a05eebd2a3f3593b48c6bdea5f7e21e5df1c138
if __name__ == "__main__":
    main()
