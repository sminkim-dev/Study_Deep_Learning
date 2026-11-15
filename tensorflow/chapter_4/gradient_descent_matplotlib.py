import func_data as func
import numpy as np
import matplotlib.pylab as plt

# 1. 경사 하강법을 실행하여 이동 경로(history) 받아오기
init_x = np.array([-3.0, 4.0])

# lr = 학습률
# lr = 10.0 , lr = 1e-10 결과를 비교해서 왜 학습률이 너무 커도 너무 작아도 안되는지 이해하기.
# -> 너무 크면 큰 값으로 발산해버리고, 너무 작으면 거의 갱신되지 않은 채 끝남.
history = func.gradient_descent_method_plt(func.function_2, init_x=init_x, lr=0.1, step_num=20)

# 2. 등고선(Contour)을 그리기 위한 X, Y 범위 설정
x = np.arange(-4.5, 4.5, 0.1)
y = np.arange(-4.5, 4.5, 0.1)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2  # function_2의 수식

# 3. 그래프 그리기
plt.figure()

# 등고선 그리기
plt.contour(X, Y, Z, colors='black')
plt.xlabel("x0")
plt.ylabel("x1")

# 4. 갱신된 경로를 선과 점으로 표시하기
# history[:, 0]은 x0 좌표들, history[:, 1]은 x1 좌표들입니다.
plt.plot(history[:, 0], history[:, 1], 'o-', color='red')

# 그래프 범위 설정 및 출력
plt.plot(0, 0, '+') # 최솟값 지점 (0, 0) 표시
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()