# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

# ---------------------------------------------------------------
# GPU(CUDA) 사용 설정
#   CuPy가 설치되어 있고 GPU가 인식되면 CuPy(GPU)를 사용하고,
#   그렇지 않으면 NumPy(CPU)로 자동 전환한다.
#   설치:  pip install cupy-cuda12x   (CUDA 12.x 드라이버 기준)
# ---------------------------------------------------------------
try:
    import cupy as xp
    xp.cuda.runtime.getDeviceCount()   # GPU가 실제로 잡히는지 확인
    GPU = True
except Exception:
    import numpy as xp
    GPU = False


# common.functions 의 함수들은 NumPy 전용이므로 xp(CuPy/NumPy) 버전으로 다시 정의
def sigmoid(x):
    return 1 / (1 + xp.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return (1.0 - s) * s


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)  # 오버플로 방지
    e = xp.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 레이블이 원-핫 벡터면 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size

# ------------------------------------------------------------------------
# 2. 수치 미분 -- GPU version, 두군데 수정 from Iris분류WithoutClass-0.py
# ------------------------------------------------------------------------
def _numerical_gradient_1d(f, x):  # p.127
    h = 1e-4
    grad = xp.zeros_like(x)  # 수정 np. --> xp.
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = tmp + h
        fxh1 = f(x)
        x[idx] = tmp - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = xp.zeros_like(X)   # 수정 np. --> xp.
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(f, x)
        return grad


# -------------------------------
"""CuPy/NumPy 겸용 수치 미분 (np.nditer는 CuPy에서 동작하지 않으므로 직접 구현).
    원소마다 루프를 돌기 때문에 GPU에서도 매우 느리다. 기울기 검증용으로만 사용할 것."""

""" def numerical_gradient(f, x):
    
    h = 1e-4
    grad = xp.zeros_like(x)
    flat_x = x.reshape(-1)      # 연속 메모리 배열이면 view가 반환됨
    flat_g = grad.reshape(-1)

    for idx in range(flat_x.size):
        tmp_val = float(flat_x[idx])
        flat_x[idx] = tmp_val + h
        fxh1 = float(f(x))  # f(x+h)

        flat_x[idx] = tmp_val - h
        fxh2 = float(f(x))  # f(x-h)

        flat_g[idx] = (fxh1 - fxh2) / (2 * h)
        flat_x[idx] = tmp_val  # 값 복원

    return grad """


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01, dtype=xp.float32):
        # RTX 4070 같은 지포스 GPU는 float64 연산이 매우 느리므로 기본값을 float32로 둔다.
        # (수치 미분으로 기울기 검증할 때는 dtype=xp.float64 권장)
        self.dtype = dtype

        # 가중치 초기화 (GPU 메모리에 바로 생성)
        self.params = {}
        self.params['W1'] = (weight_init_std * xp.random.randn(input_size, hidden_size)).astype(dtype)
        self.params['b1'] = xp.zeros(hidden_size, dtype=dtype)
        self.params['W2'] = (weight_init_std * xp.random.randn(hidden_size, output_size)).astype(dtype)
        self.params['b2'] = xp.zeros(output_size, dtype=dtype)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = xp.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = xp.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = xp.argmax(y, axis=1)
        t = xp.argmax(t, axis=1)

        accuracy = xp.sum(y == t) / float(x.shape[0])
        return float(accuracy)  # GPU 스칼라 → 파이썬 float

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = xp.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = xp.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = xp.dot(z1.T, dy)
        grads['b2'] = xp.sum(dy, axis=0)

        da1 = xp.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = xp.dot(x.T, dz1)
        grads['b1'] = xp.sum(dz1, axis=0)

        return grads
