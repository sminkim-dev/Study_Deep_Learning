# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        # self.W = np.random.randn(2,3) # 정규분포로 초기화
        self.W=np.array([[0.47355232, 0.9977393, 0.84668094], [0.85557411, 0.03563661, 0.69422093]])
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

# 1) W11 : 0.47355232 -> 0.47365232 수정 후, 손실함수 f(x) 계산  -> fxh1
print(net.loss(x,t))   # 손실함수 f(x) 계산 --> 0.9280902109609267

# 1) W11 : 0.47355232 -> 0.473465232 수정 후, 손실함수 f(x) 계산 -> fxh2
print(net.loss(x,t))   # 손실함수 f(x) 계산 --> 0.9280463614466788

print((0.9280902109609267-0.9280463614466788)/(0.0001*2))  # @L/@W11  <-- (fxh1 - fxh2) / (2*h)

# 결과 ->  0.2192475712392561
#print(dW)
