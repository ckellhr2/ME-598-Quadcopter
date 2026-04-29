import numpy as np
from NonIdealDiffDrive import main as run_nonIdeal   # rename for clarity
from IdealDiffDrive import main as run_Ideal 

def run_experiments(num_runs=5):
    all_NonIdealruns = []
    all_Idealruns = []
    for i in range(num_runs):
        print(f"=== Running simulation {i+1}/{num_runs} ===")

        # Sample initial conditions for non ideal
        non_x0 = np.random.normal(2.0, 1.0)
        non_y0 = np.random.normal(-2.0, 1.0)
        non_startpos = [non_x0, non_y0, 0.0]

        # Sample initial for ideal
        ideal_x0 = np.random.normal(0.0, 1.0)
        ideal_y0 = np.random.normal(0.0, 1.0)
        ideal_startpos = [ideal_x0, ideal_y0, 0.0]
        
        # Run one simulation
        nonIdeal_log = run_nonIdeal(non_startpos)
        Ideal_log = run_Ideal(ideal_startpos)

        # Store this run's data cleanly
        Ideal_run_data = {
            "x": np.array(Ideal_log["x"]),      # shape (T, 3)
            "t": np.array(Ideal_log["t"]),      # shape (T,)
            "startpos": np.array(non_startpos)
        }

        nonIdeal_run_data = {
            "x": np.array(nonIdeal_log["x"]),      # shape (T, 3)
            "t": np.array(nonIdeal_log["t"]),      # shape (T,)
            "startpos": np.array(non_startpos)
        }

        all_NonIdealruns.append(nonIdeal_run_data)
        all_Idealruns.append(Ideal_run_data)
    return all_NonIdealruns, all_Idealruns


if __name__ == "__main__":
    num_trials = 3
    NonIdealresults,Idealresults = run_experiments(num_trials)

    # Save to disk
    np.save(f"NonIdeal_{num_trials}runs.npy", NonIdealresults, allow_pickle=True)
    np.save(f"Ideal_{num_trials}runs.npy", Idealresults, allow_pickle=True)
    print(f"Saved {num_trials}-run datasets")