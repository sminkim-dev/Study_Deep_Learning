import tensorflow as tf

print("TensorFlow version:", tf.__version__)

print("\n[장치 확인]")
print("CPU:", tf.config.list_physical_devices("CPU"))
print("GPU:", tf.config.list_physical_devices("GPU"))

print("\n[행렬 곱 테스트]")
a = tf.constant([
    [1.0, 2.0],
    [3.0, 4.0]
])

b = tf.constant([
    [5.0, 6.0],
    [7.0, 8.0]
])

result = tf.matmul(a, b)

print(result)

print("\n[연산 장치 확인]")
with tf.device("/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"):
    x = tf.random.normal([1000, 1000])
    y = tf.random.normal([1000, 1000])
    z = tf.matmul(x, y)

print("계산 완료")
print("결과 shape:", z.shape)