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


class TrainDataReader:
    def __init__(self, dataset_dir, subset='300w_224x224',rows=224, cols=224, shuffle=True, birdeye=True):
        label_dirname = dataset_dir + subset
        print (label_dirname)

        if six.PY2:
            import commands
            label_files = commands.getoutput(
                "find %s -type f | grep .pts | sort" %
                label_dirname).splitlines()
        else:
            import subprocess
            label_files = subprocess.getstatusoutput(
                "find %s -type f | grep .pts | sort" %
                label_dirname)[-1].splitlines()

        print ('---')
        print (label_files[0])
        self.label_files = label_files
        self.label_dirname = label_dirname
        self.rows = rows
        self.cols = cols
        self.index = 0
        self.subset = subset
        self.dataset_dir = dataset_dir
        self.shuffle = shuffle
        self.reset()
        print("images total number", len(label_files))

    # 读取标注
    def read_points(self,file_name=None):
        """
        Read points from .pts file.
        """
        points = []
        with open(file_name) as file:
            line_count = 0
            for line in file:
                if "version" in line or "points" in line or "{" in line or "}" in line:
                    continue
                else:
                    loc_x, loc_y = line.strip().split(sep=" ")
                    points.append([float(loc_x), float(loc_y)])
                    line_count += 1
        return points

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
            label_name = self.label_files[self.index]
            img_name = label_name.replace('.pts', '.png')

            label = self.read_points(label_name)
            label = np.array(label)
            #print(label)
            #print("------------")
            label = label.reshape(1,-1)[0]
            #print(label.shape)
            #print(label)
            img = cv2.imread(img_name)
            if img is None:
                print("load img failed:", img_name)
                self.next_img()
            else:
                break
        try:
            img   = cv2.resize(img, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        except Exception as err:
            print('warped_error: ',err)
        img   = img.transpose((2,0,1))
        return img, label, label_name

    def get_batch(self, batch_size=1):
        imgs = []
        labels = []
        names = []
        while len(imgs) < batch_size:
            img, label, label_name = self.get_img()
            imgs.append(img)
            labels.append(label)
            names.append(label_name)
            self.next_img()
        return np.array(imgs), np.array(labels), names

    def get_batch_generator(self, batch_size, total_step):
        def do_get_batch():
            for i in range(total_step):
                gc.collect() 
                try:
                    imgs, labels, names = self.get_batch(batch_size)
                except Exception as err:
                    imgs, labels, names = self.get_batch(batch_size)
                    print('Generator　异常',err)
                #print('labels.shape = ',labels.shape)
                #print('imgs.shape = ',imgs.shape)
                imgs   = imgs.astype(np.float32)
                labels = labels.astype(np.float32)
                imgs   /= 255.0
                #labels /= 224.0
                yield i, imgs, labels, names

        batches = do_get_batch()
        try:
            from prefetch_generator import BackgroundGenerator
            batches = BackgroundGenerator(batches, 10)
        except:
            print(
                "You can install 'prefetch_generator' for acceleration of data reading."
            )
        return batches