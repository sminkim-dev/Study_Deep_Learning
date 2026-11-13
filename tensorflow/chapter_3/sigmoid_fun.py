import numpy as np
import matplotlib.pyplot as plt

def sigmoid_fun(x):
    return 1 / (1 + np.exp(-x))

x = np.arange(-6.0, 6.0, 0.1)
y = sigmoid_fun(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.show()