import numpy as np

# A = np.array([1,2,3,4])
# print(A)

# print(np.ndim(A)) # 차원 수 확인
# print(A.shape) # 배열의 형상, 튜플 반환함.
# print(A.shape[0]) 

A = np.array([[1,2],[3,4]])
print(A.shape)

B = np.array([[5,6],[7,8]])
print(B.shape)

C = np.dot(A,B) # 두 행렬의 곱은 넘파이 함수 np.dot()으로 계산함.
print(C)

# np.dot()은 입력이 1차원 배열이면 벡터를 2차원 배열이면 행렬 곱을 계산함.
# 주의 사항으로는 np.dot(A,B) 와 np.dot(B,A) 계산 값이 달라질 수 있음. 

# 행렬 곱은 항상 가운데 값이 같아야함. ex [2 to 3] * [3 to 2]
# example 2,3 * 3,2
print("---------------------------------")
fisrt_arr = np.array([[1,2,3],[4,5,6]])
#print(np.ndim(fisrt_arr))
#print(fisrt_arr.shape)
second_arr = np.array([[7,8],[9,10],[10,11]])
cal_result = np.dot(fisrt_arr, second_arr)
# 다차원 배열을 곱하려면 두 행렬의 대응하는 차원의 원소 수를 일치시켜야함.
print(cal_result)

print("2차원 행렬 * 1차원 행렬 곱")
f_a = np.array([[1,2],[3,4],[5,6]]) # [3,2]
s_a = np.array([7,8]) # [1,2] 1차원
c_r = np.dot(f_a, s_a) # 1차원 행렬 X -> 1차원 벡터

# [3,2] * [2,]가 가능한 이유, 기존의 행렬곱 처럼 그냥 내적해서 계산해버림. 그러나 이 1차원 벡터값을 진짜로 행렬로 계산할 경우 문제가 됨.
# [3,2] * [1,2]가 되므로 두 행렬의 대응하는 차원의 원소 수가 일치하지 않음. 그래서 계산이 안됨. 2 != 1
print(c_r)