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

def distance_distribution_corridor(system, path):
    distances = np.zeros((system.shape[0],system.shape[1]))
    for tstep in range(system.shape[1]):
        for trial in range(system.shape[0]):
            closest_point = path[np.argmin(np.linalg.norm(path - system[trial, tstep, 0:3], axis=1))]
            distances[trial, tstep] = np.linalg.norm(system[trial, tstep, 0:3] - closest_point)
    return distances

def calculate_stats(distances, confidence_level = 0.95):
    #Calculates the Historical Conditional Value at Risk (CVaR).
    # 1. Calculate the Value at Risk (VaR)
    var = np.percentile(distances, confidence_level * 100)
    
    # 2. Identify losses that exceed the VaR (tail of the distribution)
    tail_losses = distances[distances >= var]
    
    # 3. Calculate the average of these tail losses
    cvar = tail_losses.mean()
    
    return var, cvar

def corridor_path_sine(t_final=10.0, t_dim=100):
    times = np.linspace(0, t_final, t_dim)
    path = np.zeros((t_dim, 3))
    for tstep in range(t_dim):
        x = -1.5 + 4.2 * times[tstep]
        y = -1.25 * np.cos(2.5 * np.pi * times[tstep])
        z = 1.05 + 0.35 * times[tstep] + 0.12 * np.sin(2.0 * np.pi * times[tstep])
        path[tstep,:] = [x, y, z]
    return path


# SETUP
waypoints = np.array([
        [-1.5, -1.5, 1.05],
        [-0.9,  1.35, 1.25],
        [ 0.35, -1.25, 1.45],
        [ 1.45,  1.15, 1.10],
        [ 2.45, -0.15, 1.40],
    ])

resolution = 50 # Number of points to interpolate between waypoints for plotting
confidence_level = 0.999     # Confidence level for CVaR

# Path to the directory of this file
data_dir = "/home/justi/ME-598-Quadcopter/JustinWork/Julia/data/strong_15_85_2hz_clearance_planned/"   # Path to data

nom_name = "states_nominal"
true_name = "states_true"
L1_name = "states_L1"

nom_states = np.load(data_dir + nom_name + ".npz", allow_pickle = True)
true_states = np.load(data_dir + true_name + ".npz", allow_pickle = True)
L1_states = np.load(data_dir + L1_name + ".npz", allow_pickle = True)

t = nom_states['t'] # Universal time vector
t = t*1.01/10   # Scale time vector to reflect the progress of the actual states

# MUST be arranged so that trials are on axis 0 and 12 states are on axis 1 for the code to work
nom_u = nom_states['u']
true_u = true_states['u']
L1_u = L1_states['u']

path = corridor_path_sine(t[-1], t.shape[0])
#Debugging tools
#trial = 19
#tstep = 200
#plt.figure()
#plt.scatter(path[:,0], path[:,1], label = "Corridor Centerline")
#plt.scatter(L1_u[trial,tstep,0], L1_u[trial,tstep,1], label = "L1 Sample Point")
#plt.legend()

#Trajectories plotting
plt.figure()
for trial in range(L1_u.shape[0]):
    plt.plot(L1_u[trial,:,0], L1_u[trial,:,1])
plt.plot(path[:,0], path[:,1], label = "Corridor Centerline", linestyle='dashed', color='black')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title(f"Position Trajectories")
plt.grid(True)
plt.show()

#Trajectories plotting
ax = plt.figure().add_subplot(projection='3d')
for trial in range(L1_u.shape[0]):
    plt.plot(L1_u[trial,:,0], L1_u[trial,:,1], L1_u[trial,:,2])
plt.plot(path[:,0], path[:,1], path[:,2], label = "Corridor Centerline", linestyle='dashed', color='black')
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
plt.title(f"Position Trajectories")
plt.grid(True)
ax.set_aspect('equal')
plt.show()

nom_u_dist = distance_distribution_corridor(nom_u, path)
true_u_dist = distance_distribution_corridor(true_u, path)
L1_u_dist = distance_distribution_corridor(L1_u, path)

nom_u_stats = np.zeros((2, t.shape[0]))   # VaR and CVaR for each timestep
true_u_stats = np.zeros((2, t.shape[0]))
L1_u_stats = np.zeros((2, t.shape[0]))

for tstep in range(t.shape[0]):
    nom_u_stats[:, tstep] = calculate_stats(nom_u_dist[:, tstep], confidence_level)
    true_u_stats[:, tstep] = calculate_stats(true_u_dist[:, tstep], confidence_level)
    L1_u_stats[:, tstep] = calculate_stats(L1_u_dist[:, tstep], confidence_level)

plt.figure()
#plt.plot(t, nom_u_stats[0,:], color="#FF0000", label = f"Nominal-VaR, max = {nom_u_stats[0,:].max():.2f} m")
plt.plot(t, nom_u_stats[1,:], color="#FF7070", label = f"Nominal-CVaR, max = {nom_u_stats[1,:].max():.2f} m")
#plt.plot(t, true_u_stats[0,:], color="#00FF00", label = f"True-VaR, max = {true_u_stats[0,:].max():.2f} m")
plt.plot(t, true_u_stats[1,:], color="#70FF70", label = f"True-CVaR, max = {true_u_stats[1,:].max():.2f} m")
#plt.plot(t, L1_u_stats[0,:], color="#0000FF", label = f"L1-VaR, max = {L1_u_stats[0,:].max():.2f} m")
plt.plot(t, L1_u_stats[1,:], color="#7070FF", label = f"L1-CVaR, max = {L1_u_stats[1,:].max():.2f} m")
plt.xlabel("Time (s)")
plt.ylabel(f"VaR or CVaR (m)")
#plt.title(f"Value at Risk (VaR) and Conditional Value at Risk (CVaR) Over Time")
plt.title(f"Conditional Value at Risk (CVaR) Over Time")

plt.grid(True)
plt.legend()


plt.figure()
#plt.plot(t, nom_u_stats[0,:], color="#FF0000", label = f"Nominal-VaR, max = {nom_u_stats[0,:].max():.2f} m")
#plt.plot(t, nom_u_stats[1,:], color="#FF7070", label = f"Nominal-CVaR, max = {nom_u_stats[1,:].max():.2f} m")
#plt.plot(t, true_u_stats[0,:], color="#00FF00", label = f"True-VaR, max = {true_u_stats[0,:].max():.2f} m")
#plt.plot(t, true_u_stats[1,:], color="#70FF70", label = f"True-CVaR, max = {true_u_stats[1,:].max():.2f} m")
#plt.plot(t, L1_u_stats[0,:], color="#0000FF", label = f"L1-VaR, max = {L1_u_stats[0,:].max():.2f} m")
plt.plot(t, L1_u_stats[1,:], color="#7070FF", label = f"L1-CVaR, max = {L1_u_stats[1,:].max():.2f} m")
plt.xlabel("Time (s)")
plt.ylabel(f"VaR or CVaR (m)")
#plt.title(f"Value at Risk (VaR) and Conditional Value at Risk (CVaR) Over Time")
plt.title(f"Conditional Value at Risk (CVaR) Over Time")

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


"""
#Debugging tools
trial = 13
tstep = 30
plt.figure()
plt.scatter(path[:,0], path[:,1], label = "Corridor Centerline")
plt.scatter(nom_u[trial,tstep,0], nom_u[trial,tstep,1], label = "Nominal Sample Point")
plt.legend()


#Trajectories plotting
ax = plt.figure().add_subplot(projection='3d')
for trial in range(nom_u.shape[0]):
    plt.plot(nom_u[trial,:,0], nom_u[trial,:,1], nom_u[trial,:,2])
plt.plot(path[:,0], path[:,1], path[:,2], label = "Corridor Centerline", linestyle='dashed', color='black')
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
