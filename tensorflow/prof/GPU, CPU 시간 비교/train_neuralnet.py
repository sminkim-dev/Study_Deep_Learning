# coding: utf-8
# CPU(NumPy) 학습 시간 vs GPU(CuPy) 학습 시간 비교
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet, GPU_AVAILABLE, cp

import time

# ---------------------------------------------------------------
# 하이퍼파라미터 (CPU/GPU 모두 같은 조건으로 학습)
#   hidden_size, batch_size를 키울수록 GPU의 이점이 커진다.
#   예) hidden_size=1000, batch_size=1000 으로 바꿔서 비교해 보기
# ---------------------------------------------------------------
iters_num = 10000
batch_size = 100
hidden_size = 50
learning_rate = 0.1
seed = 0


def sync(xp):
    """GPU는 비동기로 동작하므로, 시간을 잴 때는 GPU 작업이 끝날 때까지 기다려야 한다."""
    if xp is not np:
        xp.cuda.Stream.null.synchronize()


def train(xp, label, data):
    print('=' * 60)
    print(label, '학습 시작')

    (x_train_np, t_train_np), (x_test_np, t_test_np) = data

    # ① 데이터 복사 (xp=cp 이면 CPU → GPU 메모리로 복사, xp=np 이면 그대로)
    t0 = time.perf_counter()
    x_train = xp.asarray(x_train_np, dtype=xp.float32)
    t_train = xp.asarray(t_train_np, dtype=xp.float32)
    x_test = xp.asarray(x_test_np, dtype=xp.float32)
    t_test = xp.asarray(t_test_np, dtype=xp.float32)
    sync(xp)
    copy_time = time.perf_counter() - t0

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size // batch_size, 1)

    # ② 워밍업: GPU는 첫 실행 때 커널 컴파일, cuBLAS 초기화 등으로 시간이 더 걸린다.
    #    공정한 비교를 위해 별도 신경망으로 몇 번 미리 돌려 두고 측정에서는 제외한다.
    warm = TwoLayerNet(784, hidden_size, 10, xp=xp)
    for _ in range(10):
        mask = xp.random.choice(train_size, batch_size)
        g = warm.gradient(x_train[mask], t_train[mask])
        for key in ('W1', 'b1', 'W2', 'b2'):
            warm.params[key] -= learning_rate * g[key]
        float(warm.loss(x_train[mask], t_train[mask]))
    warm.accuracy(x_test, t_test)
    sync(xp)

    # ③ 실제 학습 (같은 시드로 초기화)
    xp.random.seed(seed)
    network = TwoLayerNet(784, hidden_size, 10, xp=xp)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    eval_time = 0.0   # 정확도 계산에 쓴 시간 (학습 시간과 따로 집계)

    sync(xp)
    start = time.perf_counter()

    for i in range(iters_num):
        # 미니배치 획득
        batch_mask = xp.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 기울기 계산 (오차역전파법)
        grad = network.gradient(x_batch, t_batch)

        # 매개변수 갱신
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]  # SGD

        # 학습 경과 기록 (GPU 스칼라 → 파이썬 float)
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(float(loss))

        # 1에폭당 정확도 계산
        if i % iter_per_epoch == 0:
            e0 = time.perf_counter()
            train_acc = network.accuracy(x_train, t_train)   # 내부에서 float()로 꺼냄
            test_acc = network.accuracy(x_test, t_test)
            eval_time += time.perf_counter() - e0
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("  train acc, test acc | %.4f, %.4f" % (train_acc, test_acc))

    sync(xp)
    total_time = time.perf_counter() - start

    print('  데이터 복사 : %.3f 초' % copy_time)
    print('  전체 시간   : %.3f 초 (학습 %.3f + 정확도 계산 %.3f)'
          % (total_time, total_time - eval_time, eval_time))

    return {
        'label': label,
        'copy_time': copy_time,
        'total_time': total_time,
        'train_time': total_time - eval_time,
        'eval_time': eval_time,
        'loss': train_loss_list,
        'train_acc': train_acc_list,
        'test_acc': test_acc_list,
    }


# ---------------------------------------------------------------
# 실행
# ---------------------------------------------------------------
data = load_mnist(normalize=True, one_hot_label=True)

print('조건 : iters=%d, batch=%d, hidden=%d' % (iters_num, batch_size, hidden_size))

backends = [(np, 'CPU (NumPy)')]
if GPU_AVAILABLE:
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    backends.append((cp, 'GPU (CuPy, %s)' % gpu_name))
else:
    print('CuPy/GPU를 찾지 못해 CPU만 측정합니다.')

results = [train(xp, label, data) for xp, label in backends]

# ---------------------------------------------------------------
# 결과 비교 표
# ---------------------------------------------------------------
print('=' * 60)
print('%-30s %10s %10s %10s' % ('', '학습(초)', '정확도(초)', '전체(초)'))
for r in results:
    print('%-30s %10.3f %10.3f %10.3f'
          % (r['label'], r['train_time'], r['eval_time'], r['total_time']))

if len(results) == 2:
    cpu, gpu = results
    print('-' * 60)
    print('GPU 속도 향상 (전체) : %.2f 배' % (cpu['total_time'] / gpu['total_time']))
    print('GPU 속도 향상 (학습) : %.2f 배' % (cpu['train_time'] / gpu['train_time']))
    print('최종 test acc        : CPU %.4f / GPU %.4f'
          % (cpu['test_acc'][-1], gpu['test_acc'][-1]))

# ---------------------------------------------------------------
# 그래프: 손실 / 정확도 / 소요 시간
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for r in results:
    ax.plot(np.arange(len(r['loss'])), r['loss'], label=r['label'], alpha=0.7)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
ax.set_xlim(0, iters_num)
ax.set_ylim(0, 3.0)
ax.set_title("train loss")
ax.legend(loc='upper right')

ax = axes[1]
for r, style in zip(results, ['-', '--']):
    x = np.arange(len(r['test_acc']))
    ax.plot(x, r['train_acc'], linestyle=style, marker='o', label=r['label'] + ' train')
    ax.plot(x, r['test_acc'], linestyle=style, marker='s', label=r['label'] + ' test')
ax.set_xlabel("epochs")
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.0)
ax.set_title("accuracy")
ax.legend(loc='lower right', fontsize=8)

ax = axes[2]
labels = [r['label'].split(' ')[0] for r in results]   # 'CPU', 'GPU'
train_t = [r['train_time'] for r in results]
eval_t = [r['eval_time'] for r in results]
ax.bar(labels, train_t, label='train')
ax.bar(labels, eval_t, bottom=train_t, label='accuracy eval')
for i, r in enumerate(results):
    ax.text(i, r['total_time'], '%.2f s' % r['total_time'], ha='center', va='bottom')
ax.set_ylabel("seconds")
ax.set_title("elapsed time")
ax.legend()

plt.tight_layout()
plt.show()
