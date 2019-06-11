#-*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import time
import sys
import argparse
import paddle
import paddle.fluid as fluid
import cv2
from collections import Counter
from model.alexnet import AlexNet

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'

path = os.getcwd()

# 绘制关键点
def draw_landmark_point(image, points):
    """
    Draw landmark point on image.
    """
    for point in points:
        cv2.circle(image, (int(point[0]), int(
            point[1])), 1, (0, 255, 0), -1, cv2.LINE_AA)

def create_model(model='',image_shape=[224,224],class_num=9):

    train_image = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    
    predict = AlexNet().net(train_image)
    print('train_image.shape = ',train_image.shape)
    return predict

def load_model(exe,program,model=''):
    if model == 'ResNet':
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/ResNet.params', main_program=program)


cap = cv2.VideoCapture(0)
# 获取训练好的人脸的参数数据，这里直接从GitHub上使用默认值
face_cascade = cv2.CascadeClassifier(r'./haarcascade_frontalface_default.xml')


def infer(model):

    predict = create_model(model=model)
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program())
    load_model(exe,fluid.default_main_program(),model=model)
    print("load model succeed")


    while True:
        ret,frame = cap.read()
        cv2.imshow('frame',frame)#一个窗口用以显示原视频
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # 探测图片中的人脸
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.5,
            minNeighbors = 5,
            minSize = (5,5)
        )
        for (x, y, w, h) in faces:
            w = w*1.1
            h = h*1.3
            w = int(w)
            h = int(h)
            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            xmin = x
            xmax = x + w
            ymin = y
            ymax = y + h
            slip_image = frame[ymin:ymax,xmin:xmax]
            slip_image_224   = cv2.resize(slip_image, (224,224), interpolation=cv2.INTER_CUBIC)
            slip_image_224   = slip_image_224.transpose((2,0,1))
            imgs = []
            imgs.append(slip_image_224)
            imgs = np.array(imgs)
            imgs   = imgs.astype(np.float32)
            imgs   /= 255.0
            result = exe.run(fluid.default_main_program(),
                            feed={'img': imgs},
                            fetch_list=[predict])
            
            points = result[0]
            points = points.reshape(-1,2)
            draw_landmark_point(slip_image,points)
            cv2.imshow("image!",slip_image)

        cv2.imshow('origin image',frame)#一个窗口用以显示原视频
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    model = "ResNet"
    infer(model)

