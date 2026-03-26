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

    output_csv_path = os.path.join(script_dir, "nominal_trajectory_positions.csv")
    output_pickle_path = os.path.join(script_dir, "nominal_trajectory_positions.pkl")
    output_summary_path = os.path.join(script_dir, "nominal_trajectory_positions_summary.txt")

    physics_client = p.connect(p.DIRECT)
    trajectory_rows = []

    try:
        print("Starting nominal trajectory batch.")
        print(f"Trials to run: {N_TRIALS}")
        print(f"Goal position: {GOAL_POS}")

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

            for row in result["trajectory_rows"]:
                trajectory_rows.append({
                    "trial": int(trial_index),
                    "seed": int(seed),
                    "start_x": float(start_pos[0]),
                    "start_y": float(start_pos[1]),
                    "start_z": float(start_pos[2]),
                    "goal_x": float(GOAL_POS[0]),
                    "goal_y": float(GOAL_POS[1]),
                    "goal_z": float(GOAL_POS[2]),
                    "time_step": int(row["time_step"]),
                    "time_sec": float(row["time_sec"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]),
                })

            print(
                f"Completed trial {trial_index + 1}/{N_TRIALS}... "
                f"stored {len(result['trajectory_rows'])} time steps, "
                f"reached_goal={result['reached_goal']}",
                flush=True,
            )

    finally:
        if p.isConnected(physics_client):
            p.disconnect(physics_client)

    trajectory_df = pd.DataFrame(trajectory_rows)
    trajectory_df.to_csv(output_csv_path, index=False)
    trajectory_df.to_pickle(output_pickle_path)

    summary_lines = [
        "Nominal trajectory batch complete.",
        f"Trials: {trajectory_df['trial'].nunique()}",
        f"Stored rows: {len(trajectory_df)}",
        f"Goal position: {GOAL_POS}",
        f"Max time step: {trajectory_df['time_step'].max()}",
        f"CSV saved to: {output_csv_path}",
        f"Pickle saved to: {output_pickle_path}",
    ]

    with open(output_summary_path, "w", encoding="ascii") as f:
        f.write("\n".join(summary_lines) + "\n")

    print("\nNominal trajectory batch complete.")
    print(f"Trials: {trajectory_df['trial'].nunique()}")
    print(f"Stored rows: {len(trajectory_df)}")
    print(f"CSV saved to: {output_csv_path}")
    print(f"Pickle saved to: {output_pickle_path}")
    print(f"Summary saved to: {output_summary_path}")


if __name__ == "__main__":
    main()
