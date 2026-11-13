import numpy as np


# def step_fun(x):
#     if x>0:
#         return 1
#     else:
#         return 0
# 행렬/배열을 if 조건문에 넣으면 진리값이 모호하다고 에러남. if문은 스텔라 값을 받아들이는데 넣은 값은 벡터이기 때문.

x = np.array([-5, 0, 3, 1])

y = x > 0
print(x)

y = y.astype(int) #y = y.astype(int) # y.astype(np.int) 현재에는 해당 np.int는 사라짐.
print(y)
