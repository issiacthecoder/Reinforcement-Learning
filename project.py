# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import random

### Step 1: Initialize the problem parameters.
num_anchor_nodes = 5
total_steps = 50

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

# Loop through the epsilon values
for x in epsilons:
    current_epsilon = x
    # Initializing the 'position_estimate' to 'position_initial_estimate'
    position_estimate = position_initial_estimate.copy()

    # Initialize action counts for each epsilon
    actioncount = 0

    # Initialize Q-values for each epsilon
    reward = 0

    # Main loop for the epsilon-greedy bandit algorithm
    for y in range(total_steps):
        
        # Select three anchor nodes (action A)
        selected_positions = np.array([[0, 0, 0], [0, 0, 0],[0, 0, 0]])
        for i in range(3):
            selected_positions[i] = anchor_positions[np.random.choice(3, 1, replace=True) ]
        # Exploration: Choose random actions
        randomuniform = np.random.uniform(0, 1)
        if randomuniform < current_epsilon:
            #print("Explore!")
            filler = 5
        # Exploitation: Choose actions with highest Q-values
        else:
            # print("Exploit!")
            buffer = 1
        # Code for determining pseudoranges

        pseudoranges = [euclidean_distance(selected_positions[i], position_estimate) + np.random.uniform(-0.0001, 0.0001, 1)[0] for i in range(3)]
        pseudoranges = np.array(pseudoranges)
        print(selected_positions)
        jacobian = (target_position - selected_positions) / ((2 * pseudoranges) ** 2)
        print("\n",jacobian,"\n")

        # Determine the 'jacobian' matrix based on the selected anchor nodes
        calculate_gdop(jacobian)

        # Determine the 'gdop' value GDOP(A) from the calculated 'jacobian'

        # Determine the 'reward' R(A) using the 'gdop' value

        # Update action counts N(A)

        # Update Q-values Q(A)

        # Update position estimate

        # Store GDOP(A), R(A), Euclidean distance error for each step of 'total_steps'

        # Store GDOP values, rewards, Euclidean distance errors for each epsilon


### Step 3: Plot and analyze the results.

# Plot GDOP vs. Steps for each step and each epsilon

# Plot Reward vs. Steps for each step and each epsilon

# Plot Distance Error vs. Steps for each step and each epsilon

#
#def calculate_jacobian(selected_positions, position_estimate, pseudoranges):

 #   J = np.zeros((3, 3)).astype(float)

#    for i in range(3): # 3 rows, f, g, h representing the 3 selected anchor nodes

#       for j in range(3): # 3 columns, x, y, z representing the 3 coordinates of the target

#            JacobianCalc[i,j] = (-(selected_positions[i][j] - position_estimate[j]))/((2*pseudoranges[i])**2)

#    return JacobianCalc