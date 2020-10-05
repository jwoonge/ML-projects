import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data = np.loadtxt('dataset-a.txt', delimiter=',')
n = data.shape[0]

print(n)






###### RESULTS ######
# 01 Visualize the data
idx_class0 = (data[:,2]==0)
idx_class1 = (data[:,2]==1)
plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r')
plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b')
plt.legend(['class=0','class=1'])
plt.title('Training data')
plt.show()
# 02 Plot the loss curve obtained by the gradient descent

# 03 Plot the decision boundary of the obtained classifier

# 04 Plot the probability map of the obtained classifier

# 05 Compute the classification accuracy