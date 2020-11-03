import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
data = dataset.values

def compute_distance(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def compute_centroid(Z):
    return np.average(Z, axis=0)

a = np.array([[0,1],[1,1],[0,0]])
print(compute_centroid(a))







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