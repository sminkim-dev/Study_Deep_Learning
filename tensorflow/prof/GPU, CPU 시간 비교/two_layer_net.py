# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np

# ---------------------------------------------------------------
# GPU(CUDA) 사용 가능 여부 확인
#   CuPy가 설치되어 있고 GPU가 인식되면 GPU_AVAILABLE = True
#   설치:  pip install "cupy-cuda12x[ctk]"
#
#   CPU/GPU 비교를 위해 xp를 모듈에서 하나로 고정하지 않고,
#   TwoLayerNet(..., xp=np) / TwoLayerNet(..., xp=cp) 처럼 넘겨받는다.
# ---------------------------------------------------------------
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()   # GPU가 실제로 잡히는지 확인
    GPU_AVAILABLE = True
except Exception:
    cp = None
    GPU_AVAILABLE = False

# 예전 코드와의 호환용 (from two_layer_net import xp, GPU 를 쓰는 파일이 있을 경우)
xp = cp if GPU_AVAILABLE else np
GPU = GPU_AVAILABLE


def get_xp(x):
    """배열 x가 cupy 배열이면 cupy, numpy 배열이면 numpy 모듈을 돌려준다."""
    if cp is not None:
        return cp.get_array_module(x)
    return np


# common.functions 의 함수들은 NumPy 전용이므로
# 입력 배열의 종류(numpy/cupy)에 맞춰 동작하도록 다시 정의
def sigmoid(x):
    xp = get_xp(x)
    return 1 / (1 + xp.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return (1.0 - s) * s


def softmax(x):
    xp = get_xp(x)
    x = x - x.max(axis=-1, keepdims=True)  # 오버플로 방지
    e = xp.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy_error(y, t):
    xp = get_xp(y)
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 정답 레이블이 원-핫 벡터면 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -xp.sum(xp.log(y[xp.arange(batch_size), t] + 1e-7)) / batch_size


# ------------------------------------------------------------------------
# 수치 미분 -- CPU/GPU 겸용 (Iris분류WithoutClass-0.py 방식, 행 단위 처리)
# ------------------------------------------------------------------------
def _numerical_gradient_1d(f, x):  # p.127
    xp = get_xp(x)
    h = 1e-4
    grad = xp.zeros_like(x)
    for idx in range(x.size):
        # 주의: CuPy에서 x[idx]는 숫자가 아니라 x와 메모리를 공유하는 0차원 "뷰"이다.
        #       tmp = x[idx] 로 두면 x[idx]를 바꿀 때 tmp도 함께 바뀌어 값 복원이 깨진다.
        #       → float()로 값을 CPU로 꺼내 "복사본"을 보관한다.
        tmp = float(x[idx])
        x[idx] = tmp + h
        fxh1 = float(f(x))
        x[idx] = tmp - h
        fxh2 = float(f(x))
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        xp = get_xp(X)
        grad = xp.zeros_like(X)
        for idx, x in enumerate(X):   # x는 X의 한 행(뷰)이므로 x를 바꾸면 X도 바뀐다
            grad[idx] = _numerical_gradient_1d(f, x)
        return grad


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01, xp=np, dtype=np.float32):
        # xp     : np(CPU) 또는 cp(GPU)
        # dtype  : RTX 4070 같은 지포스 GPU는 float64 연산이 매우 느리므로 기본값 float32
        #          (수치 미분으로 기울기 검증할 때는 dtype=np.float64 권장)
        self.xp = xp
        self.dtype = dtype

        # 가중치 초기화 (xp=cp 이면 GPU 메모리에 바로 생성)
        self.params = {}
        self.params['W1'] = (weight_init_std * xp.random.randn(input_size, hidden_size)).astype(dtype)
        self.params['b1'] = xp.zeros(hidden_size, dtype=dtype)
        self.params['W2'] = (weight_init_std * xp.random.randn(hidden_size, output_size)).astype(dtype)
        self.params['b2'] = xp.zeros(output_size, dtype=dtype)

    def predict(self, x):
        xp = self.xp
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

        return cross_entropy_error(y, t)   # 0차원 배열 (GPU면 GPU 배열)

    def accuracy(self, x, t):
        xp = self.xp
        y = self.predict(x)   # (데이터 수, 10)
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
        xp = self.xp
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
