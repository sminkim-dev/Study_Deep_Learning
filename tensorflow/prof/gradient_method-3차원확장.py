# coding: utf-8
import numpy as np
import matplotlib.pylab as plt
from gradient_2d import numerical_gradient

np.set_printoptions(precision=10, suppress=True)

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []
    print('initial x    :', x) 
    for i in range(step_num):
        x_history.append(x.copy() )
        grad = numerical_gradient(f, x.copy())  # 왜 x 대신 x.copy()를 사용하나요.docx
        x -= lr * grad  # line:26 lr = 0.1 
        print('gradient :', grad)
        print('new x    :', x)        
    return x , np.array(x_history)


def function_2(x):
    return x[0]**2 + x[1]**2 + x[2]**2

init_x = np.array([-3.0, 4.0, 5.0])    

lr = 0.1
step_num = 200
x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
 
# 3D 공간 시각화 코드
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

# x_history의 각 차원을 X, Y, Z축에 매핑
ax.plot(x_history[:, 0], x_history[:, 1], x_history[:, 2], marker='o', markersize=3, label='Path', color='b')
ax.scatter(init_x[0], init_x[1], init_x[2], color='red', s=100, label='Start (-3, 4, 5)')
ax.scatter(0, 0, 0, color='green', s=100, label='Minimum (0, 0, 0)')

ax.set_xlabel("X0")
ax.set_ylabel("X1")
ax.set_zlabel("X2")
ax.set_title("Gradient Descent Trajectory (3D)")
ax.legend()
plt.show() 

