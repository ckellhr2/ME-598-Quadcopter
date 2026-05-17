import numpy as np
import matplotlib.pyplot as plt
import ot

# SETUP
state_list = []    # Select states to plot w2 data for
state_registry = ["X", "Y", "Z", "Vx", "Vy", "Vz", "Roll", "Pitch", "Yaw", "Roll Rate", "Pitch Rate", "Yaw Rate"]
w2_12 = True    # Whether to plot the 12D w2 data

# Path to the directory of this file
data_dir = "/home/justi/ME-598-Quadcopter/JustinWork/Julia/data/variable_thrust/"   # Path to data

nom_name = "states_nominal"
true_name = "states_true"
L1_name = "states_L1"

nom_states = np.load(data_dir + nom_name + ".npz", allow_pickle = True)
true_states = np.load(data_dir + true_name + ".npz", allow_pickle = True)
L1_states = np.load(data_dir + L1_name + ".npz", allow_pickle = True)

t = nom_states['t'] # Universal time vector

# MUST be arranged so that trials are on axis 0 and 12 states are on axis 1 for the code to work
nom_u = nom_states['u']
true_u = true_states['u']
L1_u = L1_states['u']

w2_nt = np.zeros((nom_u.shape[2],t.shape[0]))
w2_nl = np.zeros(w2_nt.shape)
w2_nt12 = np.zeros(t.shape[0])
w2_nl12 = np.zeros(t.shape[0])
p = 2

a = np.ones(nom_u.shape[0]) / nom_u.shape[0]
b = np.ones(true_u.shape[0]) / true_u.shape[0]
c = np.ones(L1_u.shape[0]) / L1_u.shape[0]

# Wasserstein distance between nominal and true (nt) and nominal and L1 (nl) systems at each timestep for each state and for all states together
for tstep in range(0,nom_u.shape[1]):
    if state_list != []:
        for state in range(0,nom_u.shape[2]):   # 2-Wasserstein distance per state
            w2_nt[state, tstep] = ot.wasserstein_1d(nom_u[:,tstep,state], true_u[:,tstep,state], p=p)
            w2_nl[state, tstep] = ot.wasserstein_1d(nom_u[:,tstep,state], L1_u[:,tstep,state], p=p)
    
    if w2_12:   # 12-D 2-Wasserstein distance across all states
        # p here refers to the type of norm and not wasserstein moment - must be 2
        M_nt = ot.dist(nom_u[:,tstep,:], true_u[:,tstep,:], metric='sqeuclidean', p=2)
        M_nl = ot.dist(nom_u[:,tstep,:], L1_u[:,tstep,:], metric='sqeuclidean', p=2)
        w2_nt12[tstep] = np.sqrt(ot.emd2(a, b, M_nt))
        w2_nl12[tstep] = np.sqrt(ot.emd2(a, c, M_nl))

if state_list != []:
    for state in state_list:
        plt.figure()
        plt.plot(t, w2_nt[state,:], label = "Nominal iLQR vs. Stochastic iLQR")
        plt.plot(t, w2_nl[state,:], label = "Nominal iLQR vs. Stochastic iLQR+L1")
        plt.xlabel("Time (s)")
        plt.ylabel(f"{p}-Wasserstein Distance (m)")
        plt.title(f"Comparison of {p}-Wasserstein Distances: State {state_registry[state]}")
        plt.grid(True)
        plt.legend()

if w2_12:
    plt.figure()
    plt.plot(t, w2_nt12, label = "Nominal iLQR vs. Stochastic iLQR")
    plt.plot(t, w2_nl12, label = "Nominal iLQR vs. Stochastic iLQR+L1")
    plt.xlabel("Time (s)")
    plt.ylabel(f"12D {p}-Wasserstein Distance")
    plt.title(f"Comparison of 12-Dimensional {p}-Wasserstein Distances: All States")
    plt.grid(True)
    plt.legend()
plt.show()