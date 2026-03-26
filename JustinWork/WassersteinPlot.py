# Plots the Wasserstein distance curve previously calculated

import numpy as np
import matplotlib.pyplot as plt

data = np.load('JustinWork/Wasserstein1_100runs.npy', allow_pickle = True)  # Selected file must contain Wasserstein values and time values

w1 = data[()]['w1']     # Wasserstein values
t = data[()]['t']       # Time values
p = 1                   # Specify Wasserstein moment

plt.figure(figsize=(8, 5))
plt.plot(t, w1, color="#1f77b4", linestyle="-", linewidth=1)
plt.title("Wasserstein " + str(p) + "-distance vs time")
plt.xlabel("time (s)")
plt.ylabel("W" + str(p) + " distance")
plt.grid(True)
plt.show()

lolvar = 1