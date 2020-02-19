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
from scipy.integrate import simps
from matplotlib import pyplot as plt


from model.mobilenetv2 import build_model

from data.WLFW import WLFWDataReader
import data.WLFW

from loss.pfld_loss import Loss

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'


pretrain_model = 1 #1 means pretrain_model
epochs = 3
total_step = 150000 * epochs
path = os.getcwd()


def create_reader(rows=224,cols=224):

    train_dataset = WLFWDataReader("data/train_data/list.txt")
    test_dataset  = WLFWDataReader("data/test_data/list.txt")
    return train_dataset,test_dataset

def create_model(model='',image_shape=[112,112],class_num=98):

    img = fluid.layers.data(name='img', shape=[3] + image_shape, dtype='float32')
    landmark = fluid.layers.data(name='landmark', shape=[196],dtype='float32')
    attribute = fluid.layers.data(name='attribute', shape=[6],dtype='float32')
    euler_angle = fluid.layers.data(name='euler_angle', shape=[3],dtype='float32')
    
    landmarks_pre,angles_pre = build_model(img)
    weighted_loss, loss = Loss().PFLDLoss(attribute, landmark, euler_angle, angles_pre, landmarks_pre, 250)

    print('img.shape = ',img.shape)
    print('landmark.shape = ',landmark.shape)
    print('euler_angle.shape = ',euler_angle.shape)
    print('attribute.shape = ',attribute.shape)
    
    return landmarks_pre,angles_pre,weighted_loss,loss

def load_model(exe,program,model=''):
    if model == 'mobilenetv2':
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/mobilenetv2.params', main_program=program)


def save_model(exe,program,model=''):
    if model == 'mobilenetv2':
        fluid.io.save_params(executor=exe, dirname="", filename=path+'/params/mobilenetv2.params', main_program=program)

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34 # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse

def compute_auc(errors, failureThreshold, step=0.0001, showCurve=False):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced =  [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate

def test(model,DataSet):

    landmarks_pre,angles_pre,weighted_loss,loss = create_model(model='ResNet')

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    #fluid.memory_optimize(fluid.default_main_program(),print_log=False, skip_opt_set=set([landmarks_pre.name,angles_pre.name,weighted_loss.name,loss.name]))

    if pretrain_model:
        load_model(exe,fluid.default_main_program(),model=model)
        print("load model succeed")
    else:
        print("load succeed")

    def trainLoop():
        batches = DataSet.get_batch_generator(250, 10)
        
        nme_list = []
        
        for i, imgs, landmarks_gt, attributes_gt, euler_angles_gt in batches:
            preTime = time.time()
            result = exe.run(fluid.default_main_program(),
                            feed={'img': imgs,
                                  'landmark': landmarks_gt,
                                  'attribute':attributes_gt,
                                  'euler_angle':euler_angles_gt },
                           fetch_list=[weighted_loss,loss,landmarks_pre,angles_pre]) 
            nowTime = time.time()
            
            landmarks = result[2]
            #print(landmarks)
            #print('gt',landmarks_gt.shape)
            #print('pre',landmarks.shape)
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
            landmarks_gt = landmarks_gt.reshape(landmarks_gt.shape[0], -1, 2)# landmarks_gt

            nme_temp = compute_nme(landmarks, landmarks_gt)
            for item in nme_temp:
                nme_list.append(item)
            

            if i % 2 == 0:
                print("step {:d},loss {:.6f},step_time: {:.3f}".format(
                    i,result[1][0],nowTime - preTime))
                    
                    
        fluid.io.save_inference_model(dirname=path+'/inference', feeded_var_names=['img'], target_vars=[landmarks_pre,angles_pre], executor=exe)
 
        # nme
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))

    trainLoop()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = "mobilenetv2"
    train_dataset,test_dataset = create_reader()
    test(model,test_dataset)