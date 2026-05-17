import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ot

def distance_distribution(system, obstacle_pos, radius):
    distances = np.zeros((system.shape[0],system.shape[1]))
    for tstep in range(system.shape[1]):
        for trial in range(system.shape[0]):
            distances[trial, tstep] = np.linalg.norm(system[trial, tstep, 0:3] - obstacle) - radius
    return distances

def calculate_stats(distances, confidence_level = 0.05):
    #Calculates the Historical Conditional Value at Risk (CVaR).
    # 1. Calculate the Value at Risk (VaR)
    var = np.percentile(distances, confidence_level * 100)
    
    # 2. Identify losses that exceed the VaR (tail of the distribution)
    tail_losses = distances[distances <= var]
    
    # 3. Calculate the average of these tail losses
    cvar = tail_losses.mean()
    
    return var, cvar

# SETUP
obstacle = [1,1,1]  # Obstacle position
r = 0.2             # Obstacle radius
confidence_level = 0.05     # Confidence level for CVaR


# Path to the directory of this file
data_dir = "/home/justi/ME-598-Quadcopter/JustinWork/Julia/data/baseline/"   # Path to data``

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

nom_u_dist = distance_distribution(nom_u, obstacle, r)
true_u_dist = distance_distribution(true_u, obstacle, r)
L1_u_dist = distance_distribution(L1_u, obstacle, r)
    
nom_u_stats = np.zeros((2, t.shape[0]))   # VaR and CVaR for each timestep
true_u_stats = np.zeros((2, t.shape[0]))
L1_u_stats = np.zeros((2, t.shape[0]))

for tstep in range(t.shape[0]):
    nom_u_stats[:, tstep] = calculate_stats(nom_u_dist[:, tstep], confidence_level)
    true_u_stats[:, tstep] = calculate_stats(true_u_dist[:, tstep], confidence_level)
    L1_u_stats[:, tstep] = calculate_stats(L1_u_dist[:, tstep], confidence_level)

plt.figure()
plt.plot(t, nom_u_stats[0,:], color="#FF0000", label = f"Nominal-VaR, min = {nom_u_stats[0,:].min():.2f} m")
plt.plot(t, nom_u_stats[1,:], color="#FF7070", label = f"Nominal-CVaR, min = {nom_u_stats[1,:].min():.2f} m")
plt.plot(t, true_u_stats[0,:], color="#00FF00", label = f"True-VaR, min = {true_u_stats[0,:].min():.2f} m")
plt.plot(t, true_u_stats[1,:], color="#70FF70", label = f"True-CVaR, min = {true_u_stats[1,:].min():.2f} m")
plt.plot(t, L1_u_stats[0,:], color="#0000FF", label = f"L1-VaR, min = {L1_u_stats[0,:].min():.2f} m")
plt.plot(t, L1_u_stats[1,:], color="#7070FF", label = f"L1-CVaR, min = {L1_u_stats[1,:].min():.2f} m")
plt.xlabel("Time (s)")
plt.ylabel(f"VaR or CVaR (m)")
plt.title(f"Value at Risk (VaR) and Conditional Value at Risk (CVaR) Over Time")
plt.grid(True)
plt.legend()
plt.show()

"""
# View single distance distribution in time
tstep = 30
multiplier = 100
var, cvar = calculate_stats(nom_u_dist[:,tstep], confidence_level)
sns.displot(nom_u_dist[:,tstep]*multiplier, discrete=True)  # shrink = 0.8
plt.xlabel('Die Face')
plt.title('Frequency Distribution of Dice Rolls')
plt.show()
print(f"95% VaR: {var:.2%}")
print(f"95% CVaR: {cvar:.2%}")
#"""

"""
#Trajectories plotting
ax = plt.figure().add_subplot(projection='3d')
for trial in range(nom_u.shape[0]):
    plt.plot(nom_u[trial,:,0], nom_u[trial,:,1], nom_u[trial,:,2])
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
x = r * np.outer(np.cos(u), np.sin(v)) + obstacle[0]
y = r * np.outer(np.sin(u), np.sin(v)) + obstacle[1]
z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle[2]
ax.plot_surface(x, y, z)
ax.set_xlabel("X label")
ax.set_ylabel("Y label")
ax.set_zlabel("Z label")
plt.title(f"Value at Risk (VaR) and Conditional Value at Risk (CVaR) Over Time")
plt.grid(True)
ax.set_aspect('equal')
plt.show()
#"""

lolvar = 6
