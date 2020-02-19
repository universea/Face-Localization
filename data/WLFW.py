#-*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import time
import sys
import glob
import cv2
import six
import gc
import paddle
import paddle.fluid as fluid
from collections import namedtuple
import paddle.dataset as dataset


class WLFWDataReader:
    def __init__(self, file_list, rows=224, cols=224, shuffle=False):
        self.rows = rows
        self.cols = cols
        self.index = 0
        self.label_files = None
        self.shuffle = shuffle
        
        with open(file_list, 'r') as f:
            self.label_files = f.readlines()
            

        
        print("images total number", len(self.label_files))

    def reset(self, shuffle=False):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.label_files)

    def next_img(self):
        self.index += 1
        if self.index >= len(self.label_files):
            self.reset()

    def prev_img(self):
        if self.index >= 1:
            self.index -= 1

    def get_img(self):
        while True:
            
            line = self.label_files[self.index].strip().split()
            
            img = cv2.imread(line[0])
            img   = img.transpose((2,0,1))
            landmark = np.asarray(line[1:197], dtype=np.float32)
            attribute = np.asarray(line[197:203], dtype=np.int32)
            euler_angle = np.asarray(line[203:206], dtype=np.float32)
            
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break

        return img, landmark, attribute, euler_angle

    def get_batch(self, batch_size=1):
        imgs = []
        landmarks = []
        attributes = []
        euler_angles = []
        while len(imgs) < batch_size:
            img, landmark, attribute, euler_angle = self.get_img()
            imgs.append(img)
            landmarks.append(landmark)
            attributes.append(attribute)
            euler_angles.append(euler_angle)
            self.next_img()
        return np.array(imgs), np.array(landmarks), np.array(attributes), np.array(euler_angles)

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                gc.collect() 
                try:
                    imgs, landmarks, attributes, euler_angles = self.get_batch(batch_size)
                except Exception as err:
                    imgs, landmarks, attributes, euler_angles = self.get_batch(batch_size)
                    print('Generator　异常',err)

                imgs   = imgs.astype(np.float32)
                landmarks = landmarks.astype(np.float32)
                euler_angles = euler_angles.astype(np.float32)
                attributes = attributes.astype(np.float32)
                imgs   /= 255.0
                #print('imgs.shape = ',imgs.shape)
                #print('landmarks.shape = ',landmarks.shape)
                #print('attributes.shape = ',attributes.shape)
                #print('euler_angles.shape = ',euler_angles.shape)
                yield i, imgs, landmarks, attributes, euler_angles

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches, 10)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches