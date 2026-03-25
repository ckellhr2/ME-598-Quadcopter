import os

import numpy as np
import pandas as pd
import pybullet as p

from IdealSystem import run_simulation


GOAL_POS = [1.3, 2.1, 1.4]
N_TRIALS = 100
START_BOUNDS = {
    "x": (-1.5, 1.5),
    "y": (-1.5, 1.5),
    "z": (0.8, 1.8),
}


def sample_initial_condition(rng):
    return [
        float(rng.uniform(*START_BOUNDS["x"])),
        float(rng.uniform(*START_BOUNDS["y"])),
        float(rng.uniform(*START_BOUNDS["z"])),
    ]


def main():
    base_seed = 598
    rng = np.random.default_rng(base_seed)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    physics_client = p.connect(p.DIRECT)
    results = []

    try:
        for trial_index in range(N_TRIALS):
            seed = int(rng.integers(0, 2**31 - 1))
            start_pos = sample_initial_condition(rng)

            result = run_simulation(
                start_pos=start_pos,
                x_goal_pos=GOAL_POS,
                connection_mode=p.DIRECT,
                keep_alive=False,
                log_interval=0,
                verbose=False,
                reuse_existing_connection=True,
            )
            result["trial"] = int(trial_index)
            result["seed"] = int(seed)
            result["status"] = "ok"
            results.append(result)

            print(
                f"Completed {trial_index + 1}/{N_TRIALS} trials... "
                f"cost={result['trajectory_total_cost']:.6f}",
                flush=True,
            )

    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)

    df = pd.DataFrame(results).sort_values("trial").reset_index(drop=True)

    csv_path = os.path.join(script_dir, "random_initial_condition_costs.csv")
    pickle_path = os.path.join(script_dir, "random_initial_condition_costs.pkl")
    summary_path = os.path.join(script_dir, "random_initial_condition_costs_summary.txt")

    df.to_csv(csv_path, index=False)
    df.to_pickle(pickle_path)

    summary_lines = [
        "Batch complete.",
        f"Trials: {len(df)}",
        f"Goal position: {GOAL_POS}",
        f"Mean cost: {df['trajectory_total_cost'].mean():.6f}",
        f"Std cost: {df['trajectory_total_cost'].std():.6f}",
        f"Min cost: {df['trajectory_total_cost'].min():.6f}",
        f"Max cost: {df['trajectory_total_cost'].max():.6f}",
        f"Success rate: {df['reached_goal'].mean():.3f}",
        f"CSV saved to: {csv_path}",
        f"Pickle saved to: {pickle_path}",
    ]

    with open(summary_path, "w", encoding="ascii") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\nBatch complete.")
    print(f"Trials: {len(df)}")
    print(f"Mean cost: {df['trajectory_total_cost'].mean():.6f}")
    print(f"Std cost: {df['trajectory_total_cost'].std():.6f}")
    print(f"CSV saved to: {csv_path}")
    print(f"Pickle saved to: {pickle_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
