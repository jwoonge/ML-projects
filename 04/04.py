# 0. Import library
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. Load dataset
data = np.loadtxt('dataset.txt', delimiter=',')
global n; n = data.shape[0]

# 2. Sigmoid/logistic function
def sigmoid(z):
    return 1/(1+np.exp(-z))














################
###  Results ###
################

# 1. Plot the dataset in 2D cartessian coordinate system
idx_admit = (data[:,2]==1)
idx_rejec = (data[:,2]==0)
plt.scatter(data[idx_admit,0],data[idx_admit,1], marker='+',c='r')
plt.scatter(data[idx_rejec,0],data[idx_rejec,1], c='b', s=5)
plt.legend(['Admitted','Rejected'])
plt.xlabel('Exam grade 1')
plt.ylabel('Exam grade 2')
plt.title('Training data')
plt.show()

# 2. Plot 
x_values = np.linspace(-10, 10)
plt.plot(x_values, sigmoid(x_values))
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()