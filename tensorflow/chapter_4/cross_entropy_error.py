import func_data as func
import numpy as np

# 정답이 2인 경우
t = [0,0,1,0,0,0,0,0,0,0]

# 정답일 때의 출력이 0.6인 경우로, 이때의 교차 엔트로피 오차는 약 0.51
y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(func.cross_entropy_error(np.array(y), np.array(t)))

# 정답일 때의 출력이 0.1(위의 예시보다 더 낮은)인 경우로, 이때의 교차 엔트로피 오차는 2.3
y = [0.1,0.05,0.1,0.0,0.05,0.1,0.0,0.6,0.0,0.0]
print(func.cross_entropy_error(np.array(y), np.array(t)))

# 즉, 결과(오차 값)가 더 작은 첫 번째 추정이 정답일 가능성이 더 높다고 판단하였고, 이는 앞서 오차제곱함(sse)의 판단과 일치함.