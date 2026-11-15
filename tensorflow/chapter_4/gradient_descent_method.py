import func_data as func
import numpy as np
import matplotlib.pylab as plt

init_x = np.array([-3.0, 4.0])
print(func.gradient_descent_method(func.function_2, init_x=init_x, lr=0.1, step_num=100))
# result = [-6.11110793e-10  8.14814391e-10] 거의 [0,0]에 가까운 결과