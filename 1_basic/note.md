# TensorFlow 训练笔记

## 1. 数据增强 (Data Augmentation)

数据增强是一种提高模型泛化能力的技术，通过对原始图像进行变换来创建更多的训练样本。在本项目中使用了 TensorFlow 的 ImageDataGenerator 进行以下增强：

```python
image_enhance = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 1.0,      # 缩放因子，如果是图像数据，通常使用 1.0/255
    rotation_range=45,      # 随机旋转角度范围
    width_shift_range=0.15, # 水平方向移动范围
    height_shift_range=0.15,# 垂直方向移动范围
    horizontal_flip=False,  # 是否进行水平翻转
    zoom_range=0.5,        # 缩放范围，图像随机缩放的幅度为50%
)
```

注意事项：
- 使用数据增强时，输入数据需要是4维的：(batch_size, height, width, channels)
- 对于 MNIST 数据集，需要先将数据 reshape 成正确的维度：`x_train.reshape(x_train.shape[0], 28, 28, 1)`
- 数据增强可以有效防止过拟合，提高模型的泛化能力

## 2. 断点续训 (Checkpoint)

断点续训允许保存模型训练过程中的权重，以便：
- 在训练中断后可以从上次的状态继续训练
- 保存训练过程中的最佳模型
- 方便进行模型部署

### 实现方式：

1. 配置 Checkpoint 回调：
```python
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="./checkpoint/checkpoint.ckpt",  # 保存路径
    save_weights_only=True,   # 只保存权重
    save_best_only=True,      # 只保存最佳模型
)
```

2. 加载已有的权重：
```python
if os.path.exists(checkpoint_save_path + ".index"):
    print("-------------Load Checkpoint-----------------")
    model.load_weights(checkpoint_save_path)
```

3. 在训练时使用 checkpoint：
```python
model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=5,
    validation_data=(x_test, y_test),
    validation_freq=1,
    callbacks=[checkpoint_callback],  # 将checkpoint回调加入训练过程
)
```

### 注意事项：
- checkpoint 文件通常包含 .data 和 .index 文件
- `save_best_only=True` 可以确保只保存验证集上表现最好的模型
- 使用 `save_weights_only=True` 只保存模型权重，可以减小文件大小
- 加载权重时需要确保模型结构与保存时一致

## 3. 训练策略选择

在代码中，可以通过 `use_checkpoint` 变量控制是否使用断点续训：
- 当 `use_checkpoint = True` 时，使用常规训练并启用断点续训
- 当 `use_checkpoint = False` 时，使用数据增强训练，但不保存断点

这种设计允许灵活切换训练模式，方便进行实验对比。
