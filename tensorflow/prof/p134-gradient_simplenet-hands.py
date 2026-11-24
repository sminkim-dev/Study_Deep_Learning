# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.functions import softmax
from common.gradient import numerical_gradient_2d

def cross_entropy_error(y,t):
    if y.ndim==1:  # 1차원 벡터 입력 대비.... 주석처리 후 결과 비교하기
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    #print("batch size : ", batch_size)
    delta=1e-7
    return -np.sum(t*np.log(y+delta)) /batch_size

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
p=net.predict(x)
print(p)
# [1.05414809 0.63071653 1.1328074 ]
np.argmax(p)
print(net.loss(x, t))

print("-------- 손으로 손실함수 계산 ---------")
y = softmax(p)
print("Softmax 출력: ",y)
#[0.36541271 0.23927078 0.39531651]
print(-np.log(0.39531651))


