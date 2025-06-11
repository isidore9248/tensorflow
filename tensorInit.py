import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
(
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    if physical_devices
    else None
)


def init_tensor():
    x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
    x = tf.constant([[1, 2, 3], [4, 5, 6]])
    x = tf.ones(shape=(2, 3))
    x = tf.zeros(shape=(2, 3))

    x = tf.eye(3)  # * 单位阵,参数为阶数
    x = tf.random.normal(shape=(2, 3), mean=0.0, stddev=1.0)  # * 正态分布
    x = tf.random.uniform(shape=(2, 3), minval=0, maxval=1)  # * 均匀分布
    x = tf.range(9)  # * 生成0-8的序列

    x = tf.cast(x, dtype=tf.float32)  # * 转换数据类型
    x = tf.reshape(x, shape=(3, 3))  # * 重塑形状
    print(x)


def math_tensor():
    x = tf.constant([[1, 2], [3, 4]])
    y = tf.constant([[1, 2], [3, 4]])

    z = x + y
    z = x * y  # * 对应元素相乘
    z = x @ y  # * 矩阵乘法

    # ? 可支持更高维度的乘法，axes=1 下 等价于 x @ y
    z = tf.tensordot(x, y, axes=1)  # * 张量点积

    z = tf.reduce_sum(x, axis=0)  # * 沿着指定列求和

    print(z)


if __name__ == "__main__":
    # Your code here
    print("TensorFlow version:", tf.__version__)
    init_tensor()
    math_tensor()