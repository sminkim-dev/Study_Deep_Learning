# coding: utf-8
import numpy as np
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.gradient import numerical_gradient
from common.functions import softmax, cross_entropy_error

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

y = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

#p=net.predict(x)
#t=np.array([0,0,1])
print(net.loss(y,t))  #f(x) 계산
#f = lambda w: net.loss(x, t)
def f(W) :
    return net.loss(y,t)

#dW = numerical_gradient(f, net.W)

dW = numerical_gradient(lambda w: net.loss(y, t), net.W)

print(dW)
