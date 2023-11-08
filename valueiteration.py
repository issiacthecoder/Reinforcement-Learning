import numpy as np
import math

# Define environment
environment = np.zeros((3,4))
environment[1][1] = np.nan
environment[0][3] = 1
environment[1][3] = -1
newenvironment = environment.copy()

# Initialize variables
gamma = 0.9
theta = 0.000000000000001
delta = theta
row = len(environment) # 3
col = np.size(environment,1) # 4

# Delta will iteratively increase until greater than or equal to theta
while delta >= theta:

    delta = 0
    environment = newenvironment.copy()

    # Nested loops will check each element of the array
    for x in range(row):
        for y in range(col):

            # Check if the element is a positive state, negative state, or a "blocked" element
            if environment[x][y] != "nan" and environment[x][y] != 1 and environment[x][y] != -1:
                
                neighbors = []

                # Checking upper left corner
                if x == 0 and y == 0:
                    up = environment[x][y]
                    left = environment[x][y]
                    right = environment[x][y + 1]
                    down = environment[x + 1][y]

                # Checking upper row
                if x == 0 and (y != 0 and y != np.size(environment,1) - 1):
                    up = environment[x][y]
                    left = environment[x][y - 1]
                    right = environment[x][y + 1]
                    down = environment[x + 1][y]

                # Checking upper right corner
                if x == 0 and y == np.size(environment, 1) - 1:
                    up = environment[x][y]
                    left = environment[x][y - 1]
                    right = environment[x][y]
                    down = environment[x + 1][y]

                # Checking right column
                if y == (np.size(environment, 1) - 1) and (x != 0 and x != len(environment) - 1):
                    up = environment[x][y - 1]
                    left = environment[x][y - 1]
                    right = environment[x][y]
                    down = environment[x + 1][y]

                # Checking lower right corner
                if x == len(environment) - 1 and y == np.size(environment, 1) - 1:
                    up = environment[x - 1][y]
                    left = environment[x][y - 1]
                    right = environment[x][y]
                    down = environment[x][y]

                # Checking lower row
                if x == len(environment) - 1 and (y != 0 and y != np.size(environment, 1) - 1):
                    up = environment[x - 1][y]
                    left = environment[x][y - 1]
                    right = environment[x][y + 1]
                    down = environment[x][y]

                # Checking lower left corner
                if x == len(environment) - 1 and y == 0:
                    up = environment[x][y + 1]
                    left = environment[x][y]
                    right = environment[x][y + 1]
                    down = environment[x][y]

                # Checking left column
                if y == 0 and (x != 0 and x != len(environment) - 1):
                    up = environment[x - 1][y]
                    left = environment[x][y]
                    right = environment[x][y + 1]
                    down = environment[x + 1][y]
                
                # Checking in between cells
                if y != 0 and x!= 0 and x != len(environment) - 1 and y != np.size(environment, 1) - 1:
                    up = environment[x - 1][y]
                    left = environment[x][y - 1]
                    right = environment[x][y + 1]
                    down = environment[x + 1][y]
                
                # Store the immediate neighboring state values and sort them from lowest to highest
                neighbors = [up, down, left, right]
                sorted = sort(neighbors)

                # If there is a 1, -1, or nan value present, delete it. 
                if np.isnan(sorted).any():
                    sorted = np.delete(sorted, 3) # nan is the lowest value
                if sorted[0] == -1 and sorted[-1] == 0:
                    sorted = np.delete(sorted, 0)
                
                # Calculate the reward using the bellman equation
                reward = 0.8 * ( gamma * sorted[-1])
                for i in range(len(sorted) - 1):
                    reward += 0.1 * (gamma * sorted[-i-2])

                # If the element is nan, ignore. Otherwise, update new state value
                if np.isnan(newenvironment[x][y]):
                    continue
                else:
                    newenvironment[x][y] = round(reward, 2)

                # Find the updated delta value by taking the absolute difference between the new state value and the old state value
                if delta < abs((newenvironment[x][y] - environment[x][y])):
                    delta = (newenvironment[x][y] - environment[x][y])

    print("\n", newenvironment)
environment = newenvironment.copy()
