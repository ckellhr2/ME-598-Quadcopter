import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "random_initial_condition_costs.csv")
    output_path = os.path.join(script_dir, "cost_distribution.png")

    df = pd.read_csv(csv_path)
    costs = df["trajectory_total_cost"].dropna()

    if costs.empty:
        raise RuntimeError("No valid trajectory_total_cost values found in the CSV.")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(costs, bins=20, color="#2f5d8a", edgecolor="white", alpha=0.9)
    ax.axvline(costs.mean(), color="#c0392b", linestyle="--", linewidth=2, label=f"Mean = {costs.mean():.2f}")
    ax.axvline(costs.median(), color="#1f7a1f", linestyle="-.", linewidth=2, label=f"Median = {costs.median():.2f}")

    ax.set_title("Trajectory Cost Distribution")
    ax.set_xlabel("Trajectory Total Cost")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.show()

    print(f"Loaded {len(costs)} cost values from: {csv_path}")
    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()
