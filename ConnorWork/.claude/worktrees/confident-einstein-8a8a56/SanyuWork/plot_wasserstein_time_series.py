import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot Wasserstein-1 time series from CSV")
    parser.add_argument(
        "--input-csv",
        default="wasserstein_time_series.csv",
        help="Path to the Wasserstein time series CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where plots will be saved",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario name to plot",
    )
    parser.add_argument(
        "--show-mean",
        action="store_true",
        help="Also plot the mean of x/y/z Wasserstein distances",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignoring unknown command-line args: {unknown}")

    input_csv_path = os.path.abspath(args.input_csv)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Wasserstein CSV file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)
    required_columns = {
        "scenario",
        "time_step",
        "time_sec",
        "wasserstein_x",
        "wasserstein_y",
        "wasserstein_z",
        "wasserstein_xyz_mean",
    }
    if not required_columns.issubset(df.columns):
        raise ValueError("Input CSV missing one or more required Wasserstein columns")

    scenarios = sorted(df["scenario"].unique())
    if args.scenario:
        scenarios = [scenario for scenario in scenarios if scenario == args.scenario]
        if not scenarios:
            raise ValueError(f"Scenario '{args.scenario}' not found in Wasserstein CSV")

    curve_specs = [
        ("wasserstein_x", "X", "blue"),
        ("wasserstein_y", "Y", "red"),
        ("wasserstein_z", "Z", "gold"),
    ]

    for scenario in scenarios:
        scenario_df = df[df["scenario"] == scenario].sort_values("time_sec")

        fig, ax = plt.subplots(figsize=(10, 6))

        for column, label, color in curve_specs:
            ax.plot(
                scenario_df["time_sec"],
                scenario_df[column],
                label=label,
                color=color,
                linewidth=2.0,
            )

        if args.show_mean:
            ax.plot(
                scenario_df["time_sec"],
                scenario_df["wasserstein_xyz_mean"],
                label="Mean",
                color="black",
                linewidth=2.2,
                linestyle="--",
            )

        ax.set_title(f"{scenario.replace('_', ' ').title()}: Wasserstein-1 vs Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Wasserstein-1 Distance")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.7)
        ax.legend()

        output_path = os.path.join(output_dir, f"wasserstein_{scenario}.png")
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved Wasserstein plot for {scenario} to: {output_path}")

    print(f"Generated {len(scenarios)} Wasserstein plot(s)")


if __name__ == "__main__":
    main()
