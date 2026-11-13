import numpy as np
import matplotlib.pyplot as plt

def ReLU(x):
    return np.maximum(0, x)

x = np.arange(-5, 5, 0.1) # !!! np.arrane(start, end, step)
y = ReLU(x)

plt.plot(x,y)
plt.title("ReLU_fun")

plt.xlim(-6, 6)
plt.ylim(-1 ,5.5)

plt.show()
