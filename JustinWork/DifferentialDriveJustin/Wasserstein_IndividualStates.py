# Plots the Wasserstein distance curve previously calculated


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


saveto = "JustinWork/Wasserstein1_100runsIndiv.npy"


import numpy as np
import matplotlib.pyplot as plt

data = np.load('JustinWork/Wasserstein1_100runs.npy', allow_pickle = True)  # Selected file must contain Wasserstein values and time values

idealstates = data[()]['ideal']     
nonidealstates = data[()]['nonideal']

t = data[()]['t']       # Time values
p = 1                   # Specify Wasserstein moment

n_timesteps = 2400
n_trials = 100

w1_indiv = np.zeros((2,2400))

# WASSERSTEIN DISTANCE CALCULATION (takes a few min to calculate)
for t_step in range(n_timesteps):
    w1_indiv[0,t_step] = 
    w1_indiv[1,t_step] = 



packitup = {'w1': w1,
            't': times,
            'ideal': idealstates,
            'nonideal': nonidealstates}

np.save(saveto, packitup, allow_pickle=True)

no = """
plt.figure(figsize=(8, 5))
plt.plot(t, w1, color="#1f77b4", linestyle="-", linewidth=1)
plt.title("Wasserstein " + str(p) + "-distance vs time")
plt.xlabel("time (s)")
plt.ylabel("W" + str(p) + " distance")
plt.grid(True)
plt.show()

lolvar = 1

"""