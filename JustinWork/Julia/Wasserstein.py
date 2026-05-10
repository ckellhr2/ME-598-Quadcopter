import numpy as np
import matplotlib.pyplot as plt
from ot import wasserstein_1d
from pathlib import Path

# Path to the directory of this file
data_dir = "/home/justi/ME-598-Quadcopter/JustinWork/Julia/npy_logs/"   # Path to data
current_folder = Path(__file__).parent.resolve()                        # Current folder for saving purposes

nom_name = "states_nominal"
true_name = "states_true"
L1_name = "states_L1"

nom_states = np.load(data_dir + nom_name + ".npz", allow_pickle = True)
true_states = np.load(data_dir + true_name + ".npz", allow_pickle = True)
L1_states = np.load(data_dir + L1_name + ".npz", allow_pickle = True)

t = nom_states['t'] # Universal time vector

nom_u = nom_states['u']
true_u = true_states['u']
L1_u = L1_states['u']

#nom_mean = nom_states['mean']
#true_mean = true_states['mean']
#L1_mean = L1_states['mean']

#nom_var = nom_states['var']
#true_var = true_states['var']
#L1_var = L1_states['var']

w2_nt = np.zeros((nom_u.shape[2],nom_u.shape[1]))
w2_nl = np.zeros(w2_nt.shape)
p = 2

# Wasserstein distance between nominal and true (nt) and nominal and L1 (nl) systems at each timestep for u
for state in range(0,nom_u.shape[2]):
    for tstep in range(0,nom_u.shape[1]):
        w2_nt[state, tstep] = wasserstein_1d(nom_u[:,tstep,state], true_u[:,tstep,state], p=p)
        w2_nl[state, tstep] = wasserstein_1d(nom_u[:,tstep,state], L1_u[:,tstep,state], p=p)

state_list = [0,1,2]   # Select states to plot w2 data for
state_registry = ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw", "roll_rate", "pitch_rate", "yaw_rate"]
for state in state_list:
    plt.figure()
    plt.plot(t, w2_nt[state,:], label = "Nominal-True")
    plt.plot(t, w2_nl[state,:], label = "Nominal-L1")
    plt.xlabel("Time")
    plt.ylabel("Wasserstein Distance")
    plt.title(f"Comparison of Wasserstein {p}-Distances: State {state_registry[state]}")
    plt.grid(True)
    plt.legend()
plt.show()
