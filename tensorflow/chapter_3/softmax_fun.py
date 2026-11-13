import numpy as np

# a = np.array([0.3,2.9,4.0])
# exp_a = np.exp(a) # 지수 함수

# print(exp_a)

# sum_exp_a = np.sum(exp_a)
# print(sum_exp_a)

# y = exp_a / sum_exp_a
# print(y)

# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
    
#     return y

# a = np.array([0.3,2.9,4.0])
# print("softmax_fun(0.3 , 2.9 , 4.0) > ", end="")
# print(softmax(a))

# softmax함수 사용 시 주의할 점. e는 큰 수를 다룸. 컴퓨터는 4, 8바이트를 넘어가면 오버플로가 생김.
# 컴퓨터 수치 계산시 주의해아 하는 점이 오버플로임. 때문에 아래 식은 제대로 계산되지 않는 예를 보임.

# a = np.array([1010, 1000, 990])
# print(np.exp(a) / np.sum(np.exp(a))) # nan = not a number의 약자.

# print("-----------------")
# c = np.max(a)
# print(a-c)

# print(np.exp(a-c) / np.sum(np.exp(a - c)))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

a = np.array([0.3 , 2.9 , 4.0])
print("result : ", end="")
print(softmax(a))