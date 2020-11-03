import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
data = dataset.values

def compute_distance(a, b):
    return np.sqrt(np.sum(np.square(a-b), axis=-1))

def compute_centroid(Z):
    return np.average(Z, axis=0)

def compute_label(z, M):
    return np.argmin(compute_distance(z, M))

z = [2,2]
M = [[1,1],[2,2],[3,3]]
z = np.array(z); M = np.array(M)
print(compute_distance(z, M))
print(compute_label(z, M))
print(compute_distance(z, np.array([3,3])))





'''
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
'''