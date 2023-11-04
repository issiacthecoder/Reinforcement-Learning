import numpy as np

# Create a 3x3 NumPy array (you can use your own array)
array = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# Get the dimensions of the array
rows, cols = array.shape

# Iterate through the array
for row in range(rows):
    for col in range(cols):
        current_value = array[row, col]
        neighbors = []

        # Check and add neighboring values if they exist
        if row > 0:
            neighbors.append(array[row - 1, col])  # Top neighbor
        if row < rows - 1:
            neighbors.append(array[row + 1, col])  # Bottom neighbor
        if col > 0:
            neighbors.append(array[row, col - 1])  # Left neighbor
        if col < cols - 1:
            neighbors.append(array[row, col + 1])  # Right neighbor

        # Find the maximum neighbor
        max_neighbor = max(neighbors) if neighbors else None


print(neighbors)