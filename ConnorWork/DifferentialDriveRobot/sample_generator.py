import numpy as np
from NonIdealDiffDrive import main as run_simulation   # rename for clarity

def run_experiments(num_runs=100):
    all_runs = []

    for i in range(num_runs):
        print(f"=== Running simulation {i+1}/{num_runs} ===")

        # Sample initial condition
        x0 = np.random.normal(-1.0, 1.0)
        y0 = np.random.normal(-1.0, 1.0)
        startpos = [x0, y0, 0.0]

        # Run one simulation
        log = run_simulation(startpos=startpos)

        # Store this run's data cleanly
        run_data = {
            "x": np.array(log["x"]),      # shape (T, 3)
            "t": np.array(log["t"]),      # shape (T,)
            "startpos": np.array(startpos)
        }

        all_runs.append(run_data)

    return all_runs


if __name__ == "__main__":
    results = run_experiments(100)

    # Save to disk
    np.save("diffdrive_100runs.npy", results, allow_pickle=True)

    print("Saved 100-run dataset to diffdrive_100runs.npy")
