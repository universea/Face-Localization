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

from model.mobilenetv2 import build_model

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

def create_model(model='',image_shape=[112,112],class_num=98):

    img = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    
    landmarks_pre,angles_pre = build_model(img)

    print('img.shape = ',img.shape)

    
    return landmarks_pre,angles_pre

def load_model(exe,program,model=''):
    if model == 'mobilenetv2':
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/mobilenetv2.params', main_program=program)
        print('load model succeed')

def infer(model):

    #landmarks_pre,angles_pre = create_model(model='ResNet')

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)


    [inference_program, feed_target_names, fetch_targets] = (fluid.io.load_inference_model(dirname=path+'/inference', executor=exe))


    imgs = [] #work/Face-Localization/data/test_data/imgs/7_35_Basketball_playingbasketball_35_872_0.png

    img = cv2.imread("data/test_data/imgs/7_35_Basketball_playingbasketball_35_872_0.png")
    print('img.shape',img.shape)
    
    image = cv2.resize(img, (112, 112))
    
    image   = image.transpose((2,0,1))
    imgs.append(image)
    imgs = np.array(imgs)
    imgs = imgs.astype(np.float32)
    imgs /= 255.0
    
    result = exe.run(inference_program,
                    feed={feed_target_names[0]: imgs},
                    fetch_list=fetch_targets)

    pre_landmark = result[0]
    print(pre_landmark.shape)
    pre_landmark = pre_landmark.reshape(-1, 2) 
    print(pre_landmark)
    pre_landmark = pre_landmark * [112, 112]
    
    for (x, y) in pre_landmark.astype(np.int32):
        cv2.circle(img, (x, y), 1, (0, 0, 255))
    cv2.imwrite("infer.jpg", img)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = "mobilenetv2"
    #DataSet = create_reader()
    infer(model)