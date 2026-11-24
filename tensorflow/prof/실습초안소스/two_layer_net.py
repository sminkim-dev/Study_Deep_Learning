# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.functions import *
#from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) #Z(0,1)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
        
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
    # -------------------------------
    # 2. 수치 미분
    # -------------------------------
    
    def _numerical_gradient_1d(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)
        for idx in range(x.size):
            tmp = x[idx]
            x[idx] = tmp + h
            fxh1 = f(x)
            x[idx] = tmp - h
            fxh2 = f(x)
            grad[idx] = (fxh1 - fxh2) / (2 * h)
            x[idx] = tmp
        return grad

    def _numerical_gradient(self, f, X):          # ← 이름 변경
        if X.ndim == 1:
            return self._numerical_gradient_1d(f, X)
        else:
            grad = np.zeros_like(X)
            for idx, x in enumerate(X):
                grad[idx] = self._numerical_gradient_1d(f, x)
            return grad

    # ----- 신경망용 (학습 코드에서 호출하는 메서드) -----
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = self._numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = self._numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = self._numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = self._numerical_gradient(loss_W, self.params['b2'])
        return grads
