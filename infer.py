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


def infer(model):

    predict = create_model(model='ResNet')
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program())
    load_model(exe,fluid.default_main_program(),model=model)
    print("load model succeed")

    imgs = []

    image = cv2.imread("/home/airobot/Face-Localization/dataset/300w_224x224/indoor_012.png")
    image_224   = cv2.resize(image, (224,224), interpolation=cv2.INTER_CUBIC)
    image_224   = image_224.transpose((2,0,1))
    imgs.append(image_224)
    imgs = np.array(imgs)
    imgs   = imgs.astype(np.float32)
    imgs   /= 255.0
    result = exe.run(fluid.default_main_program(),
                    feed={'img': imgs},
                    fetch_list=[predict])
    
    points = result[0]
    points = points.reshape(-1,2)

    draw_landmark_point(image,points)
    cv2.imshow("image!",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = "ResNet"
    #DataSet = create_reader()
    infer(model)