# 0. Import library
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import copy

# 1. Load dataset
data = np.loadtxt('dataset.txt', delimiter=',')
global n; n = data.shape[0]

# 2. Sigmoid/logistic function
def sigmoid(z):
    return 1/(1+np.exp(-z))

# 3. Define the prediction function for the classification
def f_pred(X, w):
    return sigmoid(np.dot(X, w))

# 4. Define the classification loss function
def mse_loss(y, y_pred):
    return np.average(np.square(y - y_pred))

def ce_loss(y, y_pred):
    return -1*np.average(y*np.log(y_pred + np.exp(-64)) + (1-y)*np.log(1-y_pred + np.exp(-64)))

# 5. Define the gradient of the classification loss function
def grad_mse_loss(X, y, y_pred):
    return 2/n * np.dot(X.T, ((y_pred - y) * y_pred * (1-y_pred)))

def grad_ce_loss(X, y, y_pred):
    return 2/n * np.dot(X.T, (y_pred - y))

# 6. Implement the gradient decent algorithm
def grad_desc_mse(X, y, w_init, tau, max_iter):
    w = copy.deepcopy(w_init)
    y_pred = f_pred(X, w)
    loss = [mse_loss(y, y_pred)]
    for i in range(max_iter):
        grad = grad_mse_loss(X, y, y_pred)
        w = w - tau*grad
        y_pred = f_pred(X, w)
        loss.append(mse_loss(y, y_pred))
    return w, loss


def grad_desc_ce(X, y, w_init, tau, max_iter):
    w = copy.deepcopy(w_init)
    y_pred = f_pred(X, w)
    loss = [ce_loss(y, y_pred)]
    for i in range(max_iter):
        grad = grad_ce_loss(X, y, y_pred)
        w = w - tau*grad
        y_pred = f_pred(X, w)
        loss.append(ce_loss(y, y_pred))
    return w, loss

label = data[:,2].reshape([n,1])
X = np.insert(data,0,1,axis=1)[:,:3]
w_init = np.array([[-9.999],[0.15],[-0.175]])
w_mse, loss_mse = grad_desc_mse(X, label, w_init, 0.002, 50000)
w_ce, loss_ce = grad_desc_ce(X, label, w_init, 1e-4, 5000)

# 7. Scikit-learn logistic regession
logreg_sklearn = LogisticRegression()
logreg_sklearn.fit(data[:,:2], data[:,2])
w_sklearn = np.array([logreg_sklearn.intercept_[0], logreg_sklearn.coef_[0][0], logreg_sklearn.coef_[0][1]]).reshape([3,1])

# 8. Plot the Decision boundary
def decision_boundary(X, w, title):
    x1_min, x1_max = X[:,1].min(), X[:,1].max()
    x2_min, x2_max = X[:,2].min(), X[:,2].max()
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
    plt.title(title)
    plt.show()

# 9. Plot the probability map
def probability_map(w, title):
    num_a = 110
    grid_x1 = np.linspace(20,110,num_a); grid_x2 = np.linspace(20,110,num_a)
    score_x1, score_x2 = np.meshgrid(grid_x1, grid_x2)
    Z = np.zeros((len(grid_x1), len(grid_x2)))
    for i in range(len(grid_x1)):
        for j in range(len(grid_x2)):
            tmpX = np.array([1, grid_x1[i], grid_x2[j]]).reshape((1,3))
            predict_prob = (f_pred(tmpX, w))
            Z[j, i] = predict_prob

    cf = plt.contourf(score_x1, score_x2, Z, num_a, alpha=0.5, cmap='RdBu_r')
    cbar = plt.colorbar(cf)
    cbar.update_ticks()

    idx_admit = (data[:,2]==1)
    idx_rejec = (data[:,2]==0)
    plt.scatter(data[idx_admit,0],data[idx_admit,1], marker='+',c='r', label='Admitted')
    plt.scatter(data[idx_rejec,0],data[idx_rejec,1], c='b', s=5, label='Rejected')
    plt.legend()
    plt.xlabel('Exam grade 1')
    plt.ylabel('Exam grade 2')
    plt.title(title)
    plt.show()

###############
### Results ###
###############

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

# 3. Plot the loss curve in the course of gradient descent using the mean square error
plt.plot(loss_mse)
plt.title('loss curve using mean square error')
plt.show()

# 4. Plot the loss curve in the course of gradient descent using the cross-entropy error
plt.plot(loss_ce)
plt.title('loss curve using cross-entropy error')
plt.show()

# 5. Plot the decision boundary using the mean square error
decision_boundary(X, w_mse, 'decision boundary using mean square error')

# 6. Plot the decision boundary using the cross-entropy error
decision_boundary(X, w_ce, 'decision boundary using cross-entropy error')

# 7. Plot the decision boundary using the Scikit-learn logistic regression algorithm
decision_boundary(X, w_sklearn, 'decision boundary using Scikit-learn logreg alg.')

# 8. Plot the probability map using the mean square error
probability_map(w_mse, 'probability map using mean square error')

# 9. Plot the probability map using the cross-entropy error
probability_map(w_ce, 'probability map using cross-entropy error')
