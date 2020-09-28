# 0. Import library
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 1. Load dataset
data = np.loadtxt('dataset.txt', delimiter=',')
global n; n = data.shape[0]
















################
###  Results ###
################

idx_admit = (data[:,2]==1)
idx_rejec = (data[:,2]==0)
plt.scatter(data[idx_admit,0],data[idx_admit,1], marker='+',c='r')
plt.scatter(data[idx_rejec,0],data[idx_rejec,1], c='b', s=5)
plt.legend(['Admitted','Rejected'])
plt.show()