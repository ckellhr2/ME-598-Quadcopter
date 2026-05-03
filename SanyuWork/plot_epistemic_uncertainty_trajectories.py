import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

SUPTITLE_FONTSIZE = 18
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 15
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13


def main():
    parser = argparse.ArgumentParser(
        description="Plot matched nominal vs fixed epistemic uncertainty trajectories"
    )
    parser.add_argument(
        "--epistemic-csv",
        default="epistemic_uncertainty_trajectories.csv",
        help="Path to the epistemic uncertainty trajectory CSV",
    )
    parser.add_argument(
        "--nominal-csv",
        default="nominal_trajectory_positions.csv",
        help="Path to the nominal trajectory CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=8,
        help="Maximum number of matched trials to plot per scenario",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name to plot",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    epistemic_csv_path = os.path.abspath(args.epistemic_csv)
    nominal_csv_path = os.path.abspath(args.nominal_csv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(epistemic_csv_path):
        raise FileNotFoundError(f"Epistemic CSV file not found: {epistemic_csv_path}")
    if not os.path.exists(nominal_csv_path):
        raise FileNotFoundError(f"Nominal CSV file not found: {nominal_csv_path}")

    epistemic_df = pd.read_csv(epistemic_csv_path)
    nominal_df = pd.read_csv(nominal_csv_path)

    required_epistemic = {"scenario", "trial", "time_step", "time_sec", "x", "y", "z"}
    required_nominal = {"trial", "time_step", "time_sec", "x", "y", "z"}
    if not required_epistemic.issubset(epistemic_df.columns):
        raise ValueError("Epistemic CSV missing required columns")
    if not required_nominal.issubset(nominal_df.columns):
        raise ValueError("Nominal CSV missing required columns")

    scenarios = sorted(epistemic_df["scenario"].unique())
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario == args.scenario]
        if not scenarios:
            raise ValueError(f"Scenario '{args.scenario}' not found in epistemic CSV")

    nominal_trials = set(nominal_df["trial"].unique())
    coords = ["x", "y", "z"]
    titles = ["X position", "Y position", "Z position"]

    for scenario in scenarios:
        scenario_df = epistemic_df[epistemic_df["scenario"] == scenario]
        matched_trials = sorted(set(scenario_df["trial"].unique()) & nominal_trials)

        if scenario == "mass_only":
            trial_durations = (
                scenario_df.sort_values("time_sec")
                .groupby("trial")
                .tail(1)[["trial", "time_sec"]]
                .rename(columns={"time_sec": "max_time_sec"})
            )
            median_duration = trial_durations["max_time_sec"].median()
            duration_limit = 1.5 * median_duration
            matched_trials = [
                trial
                for trial in matched_trials
                if float(trial_durations.loc[trial_durations["trial"] == trial, "max_time_sec"].iloc[0]) <= duration_limit
            ]

        if args.max_trials is not None and args.max_trials > 0:
            matched_trials = matched_trials[: args.max_trials]

        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

        for trial in matched_trials:
            nominal_trial_df = nominal_df[nominal_df["trial"] == trial].sort_values("time_step")
            epistemic_trial_df = scenario_df[scenario_df["trial"] == trial].sort_values("time_step")

            common_time_steps = sorted(
                set(nominal_trial_df["time_step"].unique()) & set(epistemic_trial_df["time_step"].unique())
            )
            if not common_time_steps:
                continue

            nominal_trimmed = nominal_trial_df[nominal_trial_df["time_step"].isin(common_time_steps)]
            epistemic_trimmed = epistemic_trial_df[epistemic_trial_df["time_step"].isin(common_time_steps)]

            for axis_idx, coord in enumerate(coords):
                ax = axes[axis_idx]
                ax.plot(
                    nominal_trimmed["time_sec"],
                    nominal_trimmed[coord],
                    color="blue",
                    linestyle="-",
                    linewidth=1.6,
                    alpha=0.55,
                )
                ax.plot(
                    epistemic_trimmed["time_sec"],
                    epistemic_trimmed[coord],
                    color="red",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.70,
                )
                ax.set_ylabel(f"{coord} (m)", fontsize=LABEL_FONTSIZE)
                ax.set_title(titles[axis_idx], fontsize=TITLE_FONTSIZE)
                ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
                ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)

        axes[-1].set_xlabel("Time (s)", fontsize=LABEL_FONTSIZE)
        fig.suptitle(
            f"{scenario.replace('_', ' ').title()}: Nominal vs True",
            fontsize=SUPTITLE_FONTSIZE,
            y=0.975,
        )

        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", linestyle="-", linewidth=1.8, label="Nominal"),
            Line2D([0], [0], color="red", linestyle="--", linewidth=2.0, label="True"),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.935),
            ncol=2,
            fontsize=LEGEND_FONTSIZE,
        )

        fig.tight_layout(rect=[0, 0, 1, 0.90])
        output_path = os.path.join(output_dir, f"epistemic_uncertainty_{scenario}.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot for {scenario} to: {output_path}")

    print(f"Generated {len(scenarios)} epistemic uncertainty plot(s)")


if __name__ == "__main__":
    main()
