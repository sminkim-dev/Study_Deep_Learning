import func_data as func
import numpy as np

# 정답은 '2'
t = [0,0,1,0,0,0,0,0,0,0]
# example 1 : '2'일 확률이 가장 높다고 추정함. (0.6) -> ~9%
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0, 0.0]
print(func.sum_squares_error(np.array(y), np.array(t)))

# example 2 : '7' 일 확률이 가장 높다고 추정함. (0.7) -> ~59%
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(func.sum_squares_error(np.array(y), np.array(t)))