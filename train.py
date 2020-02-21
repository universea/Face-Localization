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

#from model.mobilenetv2 import build_model

from model.mobilenetv2 import build_model


from data.WLFW import WLFWDataReader
import data.WLFW
from learning_rate import exponential_with_warmup_decay
from loss.pfld_loss import Loss

os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.99'


pretrain_model = 1 #1 means pretrain_model
epochs = 300
total_step = int(150000 * epochs / 1024)
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
    
    weighted_loss, loss = Loss().PFLDLoss(attribute, landmark, euler_angle, angles_pre, landmarks_pre, 1024)
    
    #loss = Loss().wing_loss(landmark, landmarks_pre, w=10.0, epsilon=2.0, N_LANDMARK = 98)
    mse_loss =  Loss().mse_loss(landmark, landmarks_pre)
    
    avg_loss = 0.6*weighted_loss + 0.4*mse_loss
    
    print('img.shape = ',img.shape)
    print('landmark.shape = ',landmark.shape)
    print('euler_angle.shape = ',euler_angle.shape)
    print('attribute.shape = ',attribute.shape)
    
    print('loss = ',loss.shape)
    print('weighted_loss = ',weighted_loss.shape)
    return landmarks_pre,angles_pre,weighted_loss,loss,avg_loss

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

def load_model(exe,program,model=''):
    if model == 'mobilenetv2':
        pretrained_model = path+"/params/mobilenetv2"
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))
        fluid.io.load_vars(exe, pretrained_model, main_program=program,
                           predicate=if_exist)
    elif model == 'mobilenetv3':
        fluid.io.load_params(executor=exe, dirname="", filename=path+'/params/mobilenetv3.params', main_program=program)

def save_model(exe,program,model=''):
    if model == 'mobilenetv2':
        fluid.io.save_params(executor=exe, dirname=path+"/params/mobilenetv2", main_program=program)
    elif model == 'mobilenetv3':
        fluid.io.save_params(executor=exe, dirname=path+"/params/mobilenetv2", main_program=program)

def optimizer_setting(lr):
	batch_size = 1024
	iters = 150000 // batch_size
	boundaries = [i * iters  for i in [60,100,150]]
	values = [ i * lr for i in [1,0.5,0.1,0.05]]
    
	optimizer = fluid.optimizer.Adam(
	    #momentum=0.9,
		learning_rate=exponential_with_warmup_decay(
			learning_rate=lr,
			boundaries=boundaries,
			values=values,
			warmup_iter=200,
			warmup_factor=0.),
		regularization=fluid.regularizer.L2Decay(0.00001), )

	return optimizer

def train(model,DataSet):

    landmarks_pre,angles_pre,weighted_loss,loss,avg_loss = create_model(model='ResNet')

    optimizer = optimizer_setting(0.0002)
    optimizer.minimize(loss)

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
        batches = DataSet.get_batch_generator(1024, total_step)

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
            #print('gt',landmarks_gt.shape)
            #print('pre',landmarks.shape)
            landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # landmark 
            landmarks_gt = landmarks_gt.reshape(landmarks_gt.shape[0], -1, 2)# landmarks_gt

            lr = np.array(fluid.global_scope().find_var('learning_rate')
                            .get_tensor())

            if i % 1000 == 0 and i!= 0:
                print("Model saved")
                save_model(exe,fluid.default_main_program(),model=model)

            if i % 2 == 0:
                nme_list = []
                nme_temp = compute_nme(landmarks, landmarks_gt)
                for item in nme_temp:
                    nme_list.append(item)
                    
                # nme
                #print('nme: {:.4f}'.format(np.mean(nme_list)))
                # auc and failure rate
                failureThreshold = 0.1
                auc, failure_rate = compute_auc(nme_list, failureThreshold)
                #print('auc @ {:.1f} failureThreshold: {:.4f}'.format(auc,failureThreshold))
                #print('failure_rate: {:}'.format(failure_rate))
                
                print("step {:d},lr {:.6f},w_loss {:.6f},loss {:.6f},nme: {:.4f},auc {:.1f}, failure_rate: {:}, failureThreshold: {:.4f},step_time: {:.3f}".format(
                    i,lr[0],result[0][0],result[1][0],np.mean(nme_list),auc,failure_rate,failureThreshold,nowTime - preTime))

    trainLoop()


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='')
    parse.add_argument('--model', help='model name', nargs='?')
    args = parse.parse_args()
    model = "mobilenetv2"
    train_dataset,test_dataset = create_reader()
    train(model,train_dataset)