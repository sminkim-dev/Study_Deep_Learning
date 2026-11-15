import numpy as np

# 손실함수 영역
# sse
def sum_squares_error(y,t):
    return 0.5 * np.sum((y-t)**2)
#cee
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

# 왜 손실함수를 설정하는가?
# 신경망 학습에서는 최적의 매개변수 (가중치와 편향)를 탐색할 때 손실함수의 값을 가능ㅇ한 작게 하는 매개변수 값을 찾는다.
# 이때 매개변수의 미분(기울기)을 계산하고, 그 미분 값을 단서로 매개변수의 값을 서서히 갱신하는 과정을 반복함.
# 이때 그 가중치 매개변수의 손실함수의 미분이란? '가중치 매개변수의 값을 아주 조금 변화시켰을 때, 손실함수가 어떻게 변화나'라는 의미.
# 만약? 이 미분값이 음수면 그 가중치 매개변수를 양의 방향으로 변화시켜 손실 함수의 값을 줄일 수 있음.
# 그러나 미분값이 0이면 가중치 매개변수를 어느쪽으로 움직여도 손실함수의 값은 줄어들지 않음. 그래서 가중치 매개변수는 거기서 멈춤.


# 미분 영역
# 나쁜 구현 예시
def old_numerical_diff(f,x):
    h = 1e-50 # 반올림 오차 문제를 일으킴.
    return (f(x + h) -f(x) / h)

def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x + h) -f(x - h)) / (2 * h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def function_2(x):
    return x[0] ** 2 + x[1] ** 2
    # 또는 return np.sum(x ** 2)

def function_tmp1(x0):
    return x0 * x0 + 4.0 ** 2.0

def function_tmp2(x1):
    return 3.0 ** 2.0 + x1 * x1

# 기울기 gradient

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        
        #f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)

        #f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val # 값 복원
    
    return grad

# 경사 하강법
def gradient_descent_method(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# 등고선을 이용해서 그림 작성 용
def gradient_descent_method_plt(f, init_x, lr=0.1, step_num=100):
    x = init_x.copy()
    x_history = []  # 이동 경로를 담을 리스트

    for i in range(step_num):
        # 중요: x는 numpy 배열이라 그냥 넣으면 주소값이 복사되어 값이 다 바뀜
        # 반드시 x.copy()를 써서 당시의 좌표 상태를 박제해야 함.
        x_history.append(x.copy()) 
        
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
    return np.array(x_history)  # 리스트를 넘파이 배열로 변환해서 반환