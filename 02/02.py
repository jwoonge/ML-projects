import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('profit_population.txt', delimiter=',')
x_train = data[:,0]
y_train = data[:,1]




# Plot the training data points
plt.scatter(x_train, y_train)
plt.show()