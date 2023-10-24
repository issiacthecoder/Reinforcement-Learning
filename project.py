# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations

### Step 1: Initialize the problem parameters.
num_anchor_nodes = 5
total_steps = 10

# Initialize anchor node positions and target position
anchor_positions = np.array([[11, 30, 10], [5, 40, -20], [15, 40, 30], [5, 35, 20], [15, 35, -10]], dtype=float)
target_position = [10, 35, 0.1]

# Define two epsilon values
epsilons = [0.01, 0.3]

# Calculate the centroid of anchor node positions
centroid = np.mean(anchor_positions, axis=0)

# Set the initial position estimate as the centroid
position_initial_estimate = centroid

# Function to calculate Euclidean distance
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to calculate GDOP (Geometric Dilution of Precision)
def calculate_gdop(jacobian):
    G = np.linalg.inv(np.dot(jacobian.T, jacobian))
    gdop = np.sqrt(np.trace(G))
    return gdop

# Function to calculate reward based on GDOP
def calculate_reward(gdop):
    return np.sqrt(10/3) / gdop if gdop > 0 else 0

### Step 2: Implement the Bandit Algorithm.

actions = {
    '0' : anchor_positions[random.sample(range(5), 3)],
    '1' : anchor_positions[random.sample(range(5), 3)],
    '2' : anchor_positions[random.sample(range(5), 3)],
    '3' : anchor_positions[random.sample(range(5), 3)],
    '4' : anchor_positions[random.sample(range(5), 3)],
    '5' : anchor_positions[random.sample(range(5), 3)],
    '6' : anchor_positions[random.sample(range(5), 3)],
    '7' : anchor_positions[random.sample(range(5), 3)],
    '8' : anchor_positions[random.sample(range(5), 3)],
    '9' : anchor_positions[random.sample(range(5), 3)],
}

print(actions["0"])
print("\n", actions["1"])
print("\n", actions["2"])
print("\n", actions["3"])
print("\n", actions["4"])
print("\n", actions["5"])
print("\n", actions["6"])
print("\n", actions["7"])
print("\n", actions["8"])
print("\n", actions["9"])


# Loop through the epsilon values
for x in epsilons:
    # Initializing the 'position_estimate' to 'position_initial_estimate'
    position_estimate = position_initial_estimate.copy()

    # Initialize action counts for each epsilon
    actioncount = {}
    for i in actions:
        actioncount[i] = 0

    # Initialize Q-values for each epsilon
    qvalues = {}
    for i in actions:
        qvalues[i] = 0

    # Main loop for the epsilon-greedy bandit algorithm
    for y in range(total_steps):
        # Select three anchor nodes (action A)      
        # Exploration: Choose random actions
        randomuniform = np.random.uniform(0, 1)
        if randomuniform < x:
            selected_positions = random.choice(list(actions.values()))
        # Exploitation: Choose actions with highest Q-values
        else:
            highvalue = max(qvalues, key = qvalues.get)
            selected_positions = actions[highvalue]
            
        # Code for determining pseudoranges
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]
        pseudoranges = np.array(pseudoranges)
        
        # Determine the 'jacobian' matrix based on the selected anchor nodes
        jacobian = (position_estimate - selected_positions) / (pseudoranges)

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian)
        
        # Determine the 'reward' R(A) using the 'gdop' value
        reward = calculate_reward(gdop)

        # Update action counts N(A)
        actioncount[highvalue] += 1

        # Update Q-values Q(A)
        #qvalues[selected_positions] = 0


        # Update position estimate
        delta = np.dot(np.dot(np.linalg.inv(np.dot(jacobian.T, jacobian)), jacobian.T), ([euclidean_distance(selected_positions[i], position_estimate) for i in range(3)] - pseudoranges))
        position_estimate = position_estimate + delta

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'

        # Store GDOP values, rewards, Euclidean distance errors for each epsilon
### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon

# Plot Reward vs. Steps for each step and each epsilon

# Plot Distance Error vs. Steps for each step and each epsilon