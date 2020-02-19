# face-localization
人脸98关键点检测、头部姿态检测、视线焦点


#### 安装

```
conda create --name paddlepaddle python=3.7
conda activate pysot
pip install paddlepaddle-gpu
pip install numpy
pip install opencv-python
pip install six
```

#### 数据准备:
- WFLW Dataset Download：
  - WFLW Training and Testing images [[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)] [[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)]
- WFLW Face Annotations：
  - WFLW [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz)
- Steps：
  - Unzip above two packages and put them on `./data/WFLW/`
  - Move `./data/Mirror98.txt` to `./data/WFLW/WFLW_annotations`
  - Run `cd data`
  - Run `python3 SetPreparation.py`

#### 训练

```
python train.py
```

#### 测试

```
python test.py
```

#### 预测

```
python infer.py
```

#### 摄像头Demo

```
python Webcam.py
```

