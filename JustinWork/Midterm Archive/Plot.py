import numpy as np
import matplotlib.pyplot as plt

def plot_nonideal_vs_ideal(nonideal_file, ideal_file):
    line_width = 1
    nonideal_runs = np.load(nonideal_file, allow_pickle=True)
    ideal_runs = np.load(ideal_file, allow_pickle=True)

    # -----------------------------
    # x1(t) plot
    # -----------------------------
    plt.figure(figsize=(8, 5))

    for run in nonideal_runs:
        t = run["t"]
        x1 = run["x"][:, 0]
        plt.plot(t, x1, color="#d62728", linestyle="--", linewidth=line_width, alpha=0.8)

    for run in ideal_runs:
        t = run["t"]
        x1 = run["x"][:, 0]
        plt.plot(t, x1, color="#1f77b4", linestyle="-", linewidth=line_width, alpha=0.8)

    plt.title("x₁ vs time")
    plt.xlabel("time (s)")
    plt.ylabel("x₁ position")
    plt.grid(True)
    plt.plot([], [], color="#d62728", linestyle="--", linewidth=line_width, label="Non‑Ideal")
    plt.plot([], [], color="#1f77b4", linestyle="-", linewidth=line_width, label="Ideal")
    plt.legend()

    # -----------------------------
    # x2(t) plot
    # -----------------------------
    plt.figure(figsize=(8, 5))

    for run in nonideal_runs:
        t = run["t"]
        x2 = run["x"][:, 1]
        plt.plot(t, x2, color="#d62728", linestyle="--", linewidth=line_width, alpha=0.8)

    for run in ideal_runs:
        t = run["t"]
        x2 = run["x"][:, 1]
        plt.plot(t, x2, color="#1f77b4", linestyle="-", linewidth=line_width, alpha=0.8)

    plt.title("x₂ vs time")
    plt.xlabel("time (s)")
    plt.ylabel("x₂ position")
    plt.grid(True)
    plt.plot([], [], color="#d62728", linestyle="--", linewidth=line_width, label="Non‑Ideal")
    plt.plot([], [], color="#1f77b4", linestyle="-", linewidth=line_width, label="Ideal")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    plot_nonideal_vs_ideal("NonIdeal_100runs.npy", "Ideal_100runs.npy")
