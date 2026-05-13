import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ot

# SETUP
#state_list = [0,1,2]    # Select states to plot w2 data for
#state_registry = ["X", "Y", "Z", "Vx", "Vy", "Vz", "Roll", "Pitch", "Yaw", "Roll Rate", "Pitch Rate", "Yaw Rate"]
obstacle = [1,1,1]  # Obstacle position
r = 0.2             # Obstacle radius

# Path to the directory of this file
data_dir = "/home/justi/ME-598-Quadcopter/JustinWork/Julia/data/corridor_sharp/"   # Path to data

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

"""
# View single VaR and CVaR distribution in time
tstep = 30
multiplier = 100
var, cvar = calculate_stats(nom_u_dist[:,tstep], confidence_level)
sns.displot(nom_u_dist[:,tstep]*multiplier, discrete=True)  # shrink = 0.8
plt.xlabel('Die Face')
plt.title('Frequency Distribution of Dice Rolls')
plt.show()
print(f"95% VaR: {var:.2%}")
print(f"95% CVaR: {cvar:.2%}")
"""

#"""
#Trajectories plotting
ax = plt.figure().add_subplot(projection='3d')
for trial in range(nom_u.shape[0]):
    plt.plot(nom_u[trial,:,0], nom_u[trial,:,1], nom_u[trial,:,2])
#u = np.linspace(0, 2 * np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#x = r * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
#y = r * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
#z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
#ax.plot_surface(x, y, z)
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")
plt.title(f"Position Trajectories")
plt.grid(True)
ax.set_aspect('equal')
plt.show()
#"""
lolvar = 6
