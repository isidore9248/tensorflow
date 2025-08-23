import tensorflow as tf
from PIL import Image
import numpy as np
import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging (1)

train_path = "./mnist_image_label/mnist_train_jpg_60000/"
train_txt = "./mnist_image_label/mnist_train_jpg_60000.txt"
x_train_savepath = "./mnist_image_label/mnist_x_train.npy"
y_train_savepath = "./mnist_image_label/mnist_y_train.npy"

test_path = "./mnist_image_label/mnist_test_jpg_10000/"
test_txt = "./mnist_image_label/mnist_test_jpg_10000.txt"
x_test_savepath = "./mnist_image_label/mnist_x_test.npy"
y_test_savepath = "./mnist_image_label/mnist_y_test.npy"

checkpoint_save_path = "./checkpoint/checkpoint.ckpt"


def generateds(path, txt):
    f = open(txt, "r")  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = (
            content.split()
        )  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path = path + value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = np.array(img.convert("L"))  # 图片变为8位宽灰度值的np.array格式
        img = img / 255.0  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print("loading : " + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_


# 如果需要数据增强,则需要对图像进行reshape(batch_size, height, width, channels)
# 必须四维
def dataenhance(x_train):
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    image_enhance = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 1.0,  # 如为图像，分母为255时，可归至0～1
        rotation_range=45,  # 随机45度旋转
        width_shift_range=0.15,  # 宽度偏移
        height_shift_range=0.15,  # 高度偏移
        horizontal_flip=False,  # 水平翻转
        zoom_range=0.5,  # 将图像随机缩放阈量50％
    )

    image_enhance.fit(x_train)
    return image_enhance, x_train


def get_dataset():
    if (
        os.path.exists(x_train_savepath)
        and os.path.exists(y_train_savepath)
        and os.path.exists(x_test_savepath)
        and os.path.exists(y_test_savepath)
    ):
        print("-------------Load Datasets-----------------")
        x_train_save = np.load(x_train_savepath)
        y_train = np.load(y_train_savepath)
        x_test_save = np.load(x_test_savepath)
        y_test = np.load(y_test_savepath)
        x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
        x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
    else:
        print("-------------Generate Datasets-----------------")
        x_train, y_train = generateds(train_path, train_txt)
        x_test, y_test = generateds(test_path, test_txt)

        print("-------------Save Datasets-----------------")
        x_train_save = np.reshape(x_train, (len(x_train), -1))
        x_test_save = np.reshape(x_test, (len(x_test), -1))
        np.save(x_train_savepath, x_train_save)
        np.save(y_train_savepath, y_train)
        np.save(x_test_savepath, x_test_save)
        np.save(y_test_savepath, y_test)

    return x_train, y_train, x_test, y_test


def checkpoint_setting(model):
    """设置模型的checkpoint

    Args:
        model: Keras模型

    Returns:
        tuple: (model, callback) 包含模型和checkpoint回调
    """
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_weights_only=True,
        save_best_only=True,
    )

    if os.path.exists(checkpoint_save_path + ".index"):
        print("-------------Load Checkpoint-----------------")
        model.load_weights(checkpoint_save_path)

    return model, callback


# use_checkpoint = False
use_checkpoint = True

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = get_dataset()

    image_enhance, x_train = dataenhance(x_train)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["sparse_categorical_accuracy"],
    )

    if use_checkpoint == True:
        model, checkpoint_callback = checkpoint_setting(model)

        model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=5,
            validation_data=(x_test, y_test),
            validation_freq=1,
            callbacks=[checkpoint_callback],
        )
    else:
        model.fit(
            # 基础全连接网络
            #     x_train,
            #     y_train,
            #     batch_size=32,
            # 数据增强训练
            image_enhance.flow(x_train, y_train, batch_size=32),
            epochs=5,
            validation_data=(x_test, y_test),
            validation_freq=1,
        )

    model.summary()
