# * TensorFlow 训练模型的六步法

## 1. 导入必要的库
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("Initialization successful")

# 2. 准备数据集

# 加载数据集（以MNIST为例）
(train_images, train_labels), (test_images, test_labels) = (
    tf.keras.datasets.mnist.load_data()
)

# 数据预处理

"""
60000:样本数量,表示训练集中有60000张图片
28:图片的高度,每张图片有28个像素行
28:图片的宽度,每张图片有28个像素列
1:通道数,表示这是单通道的灰度图片(如果是RGB彩色图片则为3)
/255:原始图像像素值范围是 0-255(8位灰度图像),除以255后,像素值范围变为 0-1 之间的小数
这样处理后,所有的图像数据都被缩放到 [0,1] 区间内
"""
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# 标签编码
"""
整数标签转换为 one-hot 编码格式

转换前:
train_labels 原本是形状为 (60000,) 的一维数组
每个元素是 0-9 的整数，代表对应的数字类别
例如:[5, 0, 4, 1, 9, ...]

转换后:
train_labels 变为形状为 (60000, 10) 的二维数组
每个标签变成一个长度为10的二进制向量
例如:数字5变成 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

适配模型输出层:
模型最后一层 layers.Dense(10, activation="softmax") 输出10个类别的概率分布
需要标签也以相同格式(10维向量)进行匹配
损失函数要求:
使用的 categorical_crossentropy 损失函数需要 one-hot 编码格式的标签

"""
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

## 3. 构建模型
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

## 4. 编译模型
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

## 5. 训练模型
history = model.fit(
    train_images,
    train_labels,
    epochs=5,
    batch_size=64,
    validation_data=(test_images, test_labels),
)

## 6. 评估与预测
# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# 使用模型进行预测
predictions = model.predict(test_images[:5])
print(predictions)
