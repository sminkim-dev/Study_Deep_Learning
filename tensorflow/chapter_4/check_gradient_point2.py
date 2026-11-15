import func_data as func
import numpy as np

print(func.numerical_gradient(func.function_2, np.array([3.0,4.0])))
print(func.numerical_gradient(func.function_2, np.array([0.0,2.0])))
print(func.numerical_gradient(func.function_2, np.array([3.0,0.0])))