import pybullet as p
import os
import time
import pybullet_data

# Connect to PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
# Enable gravity
p.setGravity(0, 0, -9.8)

# Load plane so the quadrotor has something to hit
p.loadURDF("plane.urdf")

# Path to quadrotor URDF
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, "quadrotor", "quadrotor.urdf")

quad_id = p.loadURDF(urdf_path, [0, 0, 1])   # start 1 meter above ground

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(1/240)   # PyBullet default timestep