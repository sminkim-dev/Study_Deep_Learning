import func_data as func
import numpy as np
import matplotlib.pylab as plt

x = np.arange(0.0,20.0,0.1) # 0에서 20까지 0.1 간격의 배열 x를 만든다. 20.0은 제외
y = func.function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()

# 아래 방식이 동작하는 이유. function_1 함수에서 매개변수를 주지 않고 numerical_diff()에 넣을 수 있는 이유는?
# numerical_diff() 안에 들어가 있어서 function_1의 코드 그 자체를 numerical_diff의 매개 변수 f의 역할을 하게됨.
# 따라서 x의 매개변수를 넣지 않아도 됨. numerical_diff에서 계산되어지기 때문에, 비유하자면 function_1의 식만 빌리고 계산은 numerical_diff에서 실행함.
print(func.numerical_diff(func.function_1, 5))
print(func.numerical_diff(func.function_1, 10))