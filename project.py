# Import necessary libraries
import numpy as np
import random

### Step 1: Initialize the problem parameters.
num_anchor_nodes = 5
total_steps = 1000 

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

### Step 2: Implement the Bandit Algorithm

actions = np.zeros((10,3,3))
actions[0] = [11, 30, 10], [5, 40, -20], [15, 40, 30]
actions[1] = [11, 30, 10], [5, 40, -20], [5, 35, 20]
actions[2] = [11, 30, 10], [5, 40, -20], [15, 35, -10]
actions[3] = [11, 30, 10], [15, 40, 30], [5, 35, 20]
actions[4] = [11, 30, 10], [15, 40, 30], [15, 35, -10]
actions[5] = [11, 30, 10], [5, 35, 20], [15, 35, -10]
actions[6] = [5, 40, -20], [15, 40, 30], [5, 35, 20]
actions[7] = [5, 40, -20], [15, 40, 30], [15, 35, -10]
actions[8] = [5, 40, -20], [5, 35, 20], [15, 35, -10]
actions[9] = [15, 40, 30], [5, 35, 20], [15, 35, -10]

total_gdop = np.zeros((len(epsilons), total_steps))
total_reward = np.zeros((len(epsilons), total_steps))
total_error = np.zeros((len(epsilons), total_steps))


# Loop through the epsilon values
for x in range(len(epsilons)):
    # Initializing the 'position_estimate' to 'position_initial_estimate'
    position_estimate = position_initial_estimate.copy()

    # Initialize action counts for each epsilon
    actioncount = np.zeros((10,1))

    # Initialize Q-values for each epsilon
    qvalues = np.zeros((10,1))

    # Main loop for the epsilon-greedy bandit algorithm
    for y in range(total_steps):
        # Select three anchor nodes (action A) 

        # Exploration: Choose random actions
        randomuniform = np.random.uniform(0, 1)
        if randomuniform < epsilons[x]:
            index = random.randint(0,9)
            selected_positions = actions[index]

        # Exploitation: Choose actions with highest Q-values
        else:
            index = np.argmax(qvalues)
            selected_positions = actions[index]
            
        # Code for determining pseudoranges
        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]
        pseudoranges = np.array(pseudoranges)
        
        # Determine the 'jacobian' matrix based on the selected anchor nodes
        jacobian = (target_position - selected_positions) / (pseudoranges)

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'
        gdop = calculate_gdop(jacobian)
        
        # Determine the 'reward' R(A) using the 'gdop' value
        reward = calculate_reward(gdop)

        # Update action counts N(A)
        actioncount[index] += 1
        
        # Update Q-values Q(A)
        #qvalues[index] += ((qvalues[index] + (1 / actioncount[index]) * (reward - qvalues[index])))

        # Update position estimate
        delta = np.dot(np.dot(np.linalg.inv(np.dot(jacobian.T, jacobian)), jacobian.T), ([euclidean_distance(selected_positions[i], position_estimate) for i in range(3)] - pseudoranges))
        position_estimate += delta

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps' and for each epsilon
        total_gdop[x][y] = gdop
        total_reward[x][y] = reward
        total_error[x][y] = np.zeros((len(epsilons), total_steps))

### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon


# Plot Reward vs. Steps for each step and each epsilon

# Plot Distance Error vs. Steps for each step and each epsilon