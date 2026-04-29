# Made with 0 AI

# Make sure to either move this file into the directory of the sample data or include the proper path to the data when running this program
# ALSO: Set the save directory and filename for the Wasserstein below v

saveto = "JustinWork/Wasserstein1_100runs.npy"


import numpy as np
import scipy.stats as stat  # 1-moment Wasserstein distance function comes from this package

ideal = np.load('Ideal_100runs.npy', allow_pickle = True)           # What is a pickle lmao
nonideal = np.load('NonIdeal_100runs.npy', allow_pickle = True)



# STATE DATA EXTRACTION AND ORGANIZATION
# The code is set up so that the calculated state values are appended to the previously stored state values after each trial
# Therefore, the last entry will contain the entire x, t, and startpos values for all trials run in a batch
# This is why the data plotting takes so long, we aren't plotting 100 trials worth of data, but 5050 trials worth of mostly-redundant data!

ideal_x = ideal[99]['x']            # Isolate all ideal state, time, and startpos data from file
ideal_sp = ideal[99]['startpos']

nonideal_x = nonideal[99]['x']
nonideal_sp = nonideal[99]['startpos']


n_trials = ideal.size           # 100 trials
times = ideal[0]['t']           # Timestep count and value are common between ideal and nonideal system, must be taken from the first data entry since following entries have timestep arrays of length n*2400
n_timesteps = times.size        # 2400 time steps per trial
n_states = 2                    # Number of states to consider in Wasserstein calculation (x and y positions = first 2 states)

idealstates = np.zeros((n_trials,n_states,n_timesteps))         # Initialize the idealstates vector for computing Wasserstein distance for all timesteps
nonidealstates = np.zeros((n_trials,n_states,n_timesteps))      # Likewise the nonideal states
w1 = np.zeros(n_timesteps)                                      # This will store Wasserstein distances for plotting



# WASSERSTEIN DISTANCE CALCULATION (takes a few min to calculate)
for t_step in range(n_timesteps):       # Iterate over each timestep
    for n_trial in range(n_trials):     # Iterate over each trial
        # Calculate the state values (x,y) from each trial corresponding to the current timestep
        # idealstates and nonidealstates are 3d matrices
        # Each row holds the states for a unique trial, and there are as many rows as there are trials
        # The 3rd dimension is time
        idealstates[n_trial,:,t_step] = ideal_x[n_trial*n_timesteps + t_step, :n_states]            # Extract states from ideal trials at a certain timestep
        nonidealstates[n_trial,:,t_step] = nonideal_x[n_trial*n_timesteps + t_step, :n_states]      # Extract states from nonideal trial at a certain timestep
    
    w1[t_step] = stat.wasserstein_distance_nd(idealstates[:,:,t_step],nonidealstates[:,:,t_step])   # Insert whatever Wasserstein function is preferred



packitup = {'w1': w1,                           # Store all important data under one variable for convenience
            't': times,
            'ideal': idealstates,
            'nonideal': nonidealstates}

np.save(saveto, packitup, allow_pickle=True)    # Save data to previously specified location
