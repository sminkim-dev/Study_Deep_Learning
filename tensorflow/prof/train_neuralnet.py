# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet, xp, GPU

import time

if GPU:
    dev = xp.cuda.Device(0)
    name = xp.cuda.runtime.getDeviceProperties(dev.id)['name'].decode()
    print('GPU 사용 :', name)
else:
    print('CuPy/GPU를 찾지 못해 CPU(NumPy)로 실행합니다.')

# 데이터 읽기 (NumPy 배열로 읽은 뒤 GPU 메모리로 한 번만 복사)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
x_train = xp.asarray(x_train, dtype=xp.float32)
t_train = xp.asarray(t_train, dtype=xp.float32)
x_test = xp.asarray(x_test, dtype=xp.float32)
t_test = xp.asarray(t_test, dtype=xp.float32)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수 (정수로 맞춰야 i % iter_per_epoch 가 정확히 동작)
iter_per_epoch = max(train_size // batch_size, 1)

start = time.time()

for i in range(iters_num):
    # 미니배치 획득 (인덱스도 GPU에서 생성)
    batch_mask = xp.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # 수치 미분은 매개변수 4만여 개를 하나씩 계산하므로 GPU에서도 사실상 끝나지 않는다.
    # → 오차역전파법을 사용한다.
    grad = network.numerical_gradient(x_batch, t_batch)
    #grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]  # SGD

    # 학습 경과 기록 (GPU 스칼라 → 파이썬 float 로 변환해서 저장)
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(float(loss))

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

if GPU:
    xp.cuda.Stream.null.synchronize()  # GPU 작업이 모두 끝난 뒤 시간 측정
print('소요시간 : ', time.time() - start, '초')

# 그래프 그리기 (리스트는 이미 파이썬 float 이므로 NumPy/matplotlib 그대로 사용)
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
