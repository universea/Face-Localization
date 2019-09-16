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

from model.mobilenetv3 import build_model

from reader import TrainDataReader
import reader

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'


pretrain_model = 0 #1 means pretrain_model
total_step = 150000
path = os.getcwd()


def create_reader(rows=224,cols=224):

    LaneDataset = reader.TrainDataReader
    dataset = LaneDataset("data/data7426/", '300w_224x224',
                            rows=224, cols=224)
    return dataset

def create_model(model='',image_shape=[224,224],class_num=68):

    train_image = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    train_label = fluid.layers.data(name='label', shape=[136],dtype='float32')
    
    predict,loss = build_model(train_image,train_label,class_num,image_shape)
    print('train_image.shape = ',train_image.shape)
    print('predict.shape = ',predict.shape)
    print('train_label.shape = ',train_label.shape)

    print('loss = ',loss.shape)
    return predict,loss

def load_model(exe,program,model=''):
    if model == 'ResNet':
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/ResNet.params', main_program=program)


def save_model(exe,program,model=''):
    if model == 'ResNet':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/ResNet.params', main_program=program)

def train(model):

    predict,loss = create_model(model='ResNet')
    optimizer = fluid.optimizer.Adam(learning_rate=1e-5)
    optimizer.minimize(loss)
    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.memory_optimize(fluid.default_main_program(),print_log=False, skip_opt_set=set([loss.name,predict.name]))

    if pretrain_model:
        load_model(exe,fluid.default_main_program(),model=model)
        print("load model succeed")
    else:
        print("load succeed")

    def trainLoop():
        batches = DataSet.get_batch_generator(100, total_step)

        for i, imgs, labels, names in batches:
            preTime = time.time()
            result = exe.run(fluid.default_main_program(),
                            feed={'img': imgs,
                                    'label': labels},
                           fetch_list=[loss,predict])
            nowTime = time.time()


            if i % 1000 == 0 and i!= 0:
                print("Model saved")
                save_model(exe,fluid.default_main_program(),model=model)

            if i % 2 == 0:
                print("step {:d},loss {:.6f},step_time: {:.3f}".format(
                    i,result[0][0],nowTime - preTime))

    trainLoop()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = "ResNet"
    DataSet = create_reader()
    train(model)