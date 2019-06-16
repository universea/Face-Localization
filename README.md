# face-localization
人脸68关键点检测、头部姿态检测、视线焦点

2019-06-08：
loss函数设计问题，位置回归不准确。

#### 安装

```
conda create --name paddlepaddle python=3.7
conda activate pysot
pip install paddlepaddle-gpu
pip install numpy
pip install opencv-python
pip install six
```



#### 训练

```
python train.py
```

#### 预测

```
python infer.py
```

#### 摄像头Demo

```
python Webcam.py
```

