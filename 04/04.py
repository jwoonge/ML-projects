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
    return sigmoid(np.dot(X, w))

#X = data[(data[:,2]==1)]
#X2 = data[data[:,2]==0]

print(data.shape)
label = data[:,2].reshape([n,1])
X = np.insert(data[:,:2],0,1,axis=1)
w = np.array([[0],[0],[0]])

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
    return 2/n * np.dot(X.T, (y_pred - y * (y_pred * (1-y_pred))))
    #return 2/n * np.dot(X.T, (y_pred - y))

def grad_ce_loss(y, y_pred):
    return 2/n * np.dot(X.T, (y_pred - y))

# 6. Implement the gradient decent algorithm
import copy
def grad_desc(X, y, w_init, tau, max_iter):
    w_mse = copy.deepcopy(w_init); #ws_mse = [w_mse]
    w_ce = copy.deepcopy(w_init); #ws_ce = [w_ce] 
    y_pred_mse = f_pred(X, w_mse); y_pred_ce = f_pred(X, w_ce)
    loss_mse = [mse_loss(y, y_pred_mse)]; loss_ce = [ce_loss(y, y_pred_ce)]
    for i in range(max_iter):
        grad_mse = grad_mse_loss(y, y_pred_mse); grad_ce = grad_ce_loss(y, y_pred_ce)
        w_mse = w_mse - tau*grad_mse
        w_ce = w_ce - tau*grad_ce
        
        y_pred_mse = f_pred(X, w_mse); y_pred_ce = f_pred(X, w_ce)
        loss_mse.append(mse_loss(y, y_pred_mse))
        loss_ce.append(ce_loss(y, y_pred_ce))
    return w_mse, loss_mse, w_ce, loss_ce

label = data[:,2].reshape([n,1])
X = np.insert(data,0,1,axis=1)[:,:3]
w_init = np.array([[0],[0],[0]])
tau = 1e-4; max_iter = 500

w_mse, loss_mse, w_ce, loss_ce = grad_desc(X, label, w_init, tau, max_iter)
plt.plot(loss_mse)
plt.show()
plt.plot(loss_ce)
plt.show()
# Scikit-learn logistic regession
logreg_sklearn = LogisticRegression()
logreg_sklearn.fit(data[:,:2], label)
w_sklearn = np.array([logreg_sklearn.intercept_[0], logreg_sklearn.coef_[0][0], logreg_sklearn.coef_[0][1]]).reshape([3,1])
print(w_sklearn)
print(w_mse)
print(w_ce)

# compute values p(x) for multiple data points x
def decision_boundary(X, w):
    x1_min, x1_max = X[:,1].min(), X[:,1].max() # min and max of grade 1
    x2_min, x2_max = X[:,2].min(), X[:,2].max() # min and max of grade 2
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max)) # create meshgrid
    X2 = np.ones([np.prod(xx1.shape),3]) 
    X2[:,1] = xx1.reshape(-1)
    X2[:,2] = xx2.reshape(-1)
    p = f_pred(X2,w)
    p = p.reshape([xx1.shape[0],xx2.shape[0]])
    idx_admit = (data[:,2]==1)
    idx_rejec = (data[:,2]==0)
    plt.scatter(data[idx_admit,0],data[idx_admit,1], marker='+',c='r')
    plt.scatter(data[idx_rejec,0],data[idx_rejec,1], c='b', s=5)
    plt.contour(xx1, xx2, p, levels=[0,0.5,1])
    plt.show()
decision_boundary(X, w_sklearn)
    
###############
### Results ###
###############
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

