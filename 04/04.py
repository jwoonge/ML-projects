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

# 3. Define the prediction function for the classification
def f_pred(X, w):
    return sigmoid(np.dot(X, w.T))

X = data[(data[:,2]==1)]
X2 = data[data[:,2]==0]

label = data[:,2].reshape([n,1])
X = np.insert(data,0,1,axis=1)[:,:3]
w = np.array([[0,0,0]])

# 4. Define the classification loss function
def mse_loss(y, y_pred):
    return np.average(np.square(y - y_pred))

def ce_loss(y, y_pred):
    return np.average(-y*np.log(y_pred + np.exp(-64)) - (1-y)*np.log(1-y_pred + np.exp(-64)))

y_pred= f_pred(X, w)
print(mse_loss(label, y_pred))
print(ce_loss(label, y_pred))

# 5. Define the gradient of the classification loss function
def grad_mse_loss(y, y_pred):
    print(y_pred.shape)
    print((y_pred-y).shape)
    return 2/n * np.dot(X.T, (y_pred-y * (y_pred * (1-y_pred))))

def grad_ce_loss(y, y_pred):
    return 2/n * np.dot(X.T, (y_pred - y))

print(grad_mse_loss(label, y_pred))
print(grad_ce_loss(label, y_pred))




################
###  Results ###
################
'''
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

# 2. Plot the sigmoid function
x_values = np.linspace(-10, 10)
plt.plot(x_values, sigmoid(x_values))
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
'''
# 3. Plot the loss curve in the course of gradient descent using the mean square error


# 4. Plot the loss curve in the course of gradient descent using the cross-entropy error

# 5. Plot the decision boundary using the mean square error

# 6. Plot the decision boundary using the cross-entropy error

# 7. Plot the decision boundary using the Scikit-learn logistic regression algorithm

# 8. Plot the probability map using the mean square error

