import numpy as np
import matplotlib.pylab as plt

def step_fun(x):
    return np.array(x>0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1) # np.arange(end, start, step) 실제 x 데이터 생성
y = step_fun(x)

plt.plot(x, y)
plt.title("step_function")

plt.xlim(-6.0, 6.0) # x축 범위 설정
plt.ylim(-0.1, 1.1) # y축 범위 설정 (plt.show()로 보여지는 프레임의 크기 기준인듯 함.)

plt.xticks(np.arange(-6,7,1)) # 수치 칸 범위 제어 가능
plt.yticks(np.arange(-0.1,1.2,0.2))

plt.xlabel("X")
plt.ylabel("Y")

plt.show()