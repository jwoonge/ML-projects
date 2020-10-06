import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 1. Load dataset
data = np.loadtxt('dataset-b.txt', delimiter=',')
global n; n = data.shape[0]
global nx; nx = data.shape[1]-1




###### RESULTS ######
# 01 Visualize the data
idx_class0 = (data[:,2]==0)
idx_class1 = (data[:,2]==1)
plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r')
plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b')
plt.legend(['class=0','class=1'])
plt.title('Training data')
plt.show()