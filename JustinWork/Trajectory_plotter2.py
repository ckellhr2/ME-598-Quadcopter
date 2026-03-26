# This is a much faster and cleaner render of the data produced by the 100 trials
# It takes advantage of the fact that all the trial data is stored in the last entry of the "_runs" files, so you only need to worry about
# rendering that data and disregard the other 99 entries - this produces a cleaner picture too as you can see in the ppt

import numpy as np
import matplotlib.pyplot as plt

def plot_nonideal_vs_ideal(nonideal_file, ideal_file):

    nonideal_runs = np.load(nonideal_file, allow_pickle=True)
    ideal_runs = np.load(ideal_file, allow_pickle=True)

    # -----------------------------
    # x1(t) plot
    # -----------------------------

    plt.figure(figsize=(8, 5))
    t = nonideal_runs[-1]["t"]
    x1 = nonideal_runs[-1]["x"][:, 0]
    x2 = ideal_runs[-1]["x"][:, 0]
    plt.plot(t, x1, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8, label="Non‑Ideal")
    plt.plot(t, x2, color="#1f77b4", linestyle="-", linewidth=1.5, alpha=0.8, label="Ideal")
    plt.title("x₁ vs time")
    plt.xlabel("time (s)")
    plt.ylabel("x₁ position")
    plt.grid(True)
    plt.legend()

    # -----------------------------
    # x2(t) plot
    # -----------------------------

    plt.figure(figsize=(8, 5))
    t = nonideal_runs[-1]["t"]
    x1 = nonideal_runs[-1]["x"][:, 1]
    x2 = ideal_runs[-1]["x"][:, 1]
    plt.plot(t, x1, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8, label="Non‑Ideal")
    plt.plot(t, x2, color="#1f77b4", linestyle="-", linewidth=1.5, alpha=0.8, label="Ideal")
    plt.title("x₂ vs time")
    plt.xlabel("time (s)")
    plt.ylabel("x₂ position")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    plot_nonideal_vs_ideal("NonIdeal_100runs.npy", "Ideal_100runs.npy")
