import numpy as np
import heapq
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

# Grid size
grid_size = (101, 101, 101)

# Create random weights
weights = np.zeros(grid_size)
for _ in range(5000):  # Assign higher weights to some random points
    x, y, z = np.random.randint(0, 101, size=3)
    weights[x, y, z] = random.randint(5, 20)  # Assign random high weights

# Velocity of travel (m/s)
v = 1  # Can be changed

def heuristic(a, b):
    """Euclidean distance heuristic for A*"""
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, end, occupied_points):
    """A* search algorithm avoiding occupied points."""
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    pq = []
    heapq.heappush(pq, (0, start))
    came_from = {}
    cost_so_far = {start: 0}
    
    while pq:
        _, current = heapq.heappop(pq)
        if current == end:
            break
        
        for d in directions:
            neighbor = (current[0] + d[0], current[1] + d[1], current[2] + d[2])
            if 0 <= neighbor[0] < 101 and 0 <= neighbor[1] < 101 and 0 <= neighbor[2] < 101:
                new_cost = cost_so_far[current] + 1 + weights[neighbor]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    if neighbor not in occupied_points:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost + heuristic(neighbor, end)
                        heapq.heappush(pq, (priority, neighbor))
                        came_from[neighbor] = current
    
    # Reconstruct path
    path = []
    current = end
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    
    return path

def find_non_conflicting_paths(paths):
    """Find shortest paths while avoiding conflicts in time."""
    occupied_points = {}
    final_paths = []
    
    for start, end in paths:
        path = astar(start, end, occupied_points)
        time_step = 0
        for point in path:
            occupied_points[(point, time_step)] = True
            time_step += 1
        final_paths.append(path)
    
    return final_paths

# User input paths
input_paths = [
    ((0, 0, 0), (50, 50, 50)),
    ((10, 10, 10), (60, 60, 60)),
    ((5, 5, 5), (40, 40, 40))
]

# Compute non-conflicting paths
computed_paths = find_non_conflicting_paths(input_paths)

# Plot in 3D
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

colors = ['r', 'g', 'b', 'c', 'm', 'y']  # Assign different colors

for i, path in enumerate(computed_paths):
    xs, ys, zs = zip(*path)
    ax.plot(xs, ys, zs, color=colors[i % len(colors)], label=f'Path {i+1}')
    ax.scatter(xs[0], ys[0], zs[0], color='black', marker='o', s=100, label=f'Start {i+1}')
    ax.scatter(xs[-1], ys[-1], zs[-1], color='blue', marker='x', s=100, label=f'End {i+1}')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
plt.show()
