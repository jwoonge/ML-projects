import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
data = dataset.values

def compute_distance(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

print(compute_distance(np.array([1,3]), np.array([2,4])))








### RESULTS ###
# 01 Plot the data points
plt.scatter(data[:,0], data[:,1], label='data')
plt.title('data point')
plt.legend()
plt.show()

# 02 Visualize the initial condition of the point labels

# 03 Plot the loss curve

# 04 Plot the centroid of each cluster

# 05 Plot the final clustering result