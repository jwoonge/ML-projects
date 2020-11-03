import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data-kmeans.csv')
data = dataset.values










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