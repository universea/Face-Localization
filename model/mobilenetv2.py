#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    'MobileNetV2_x0_25', 'MobileNetV2_x0_5'
    'MobileNetV2_x0_75', 'MobileNetV2_x1_0', 'MobileNetV2_x1_5',
    'MobileNetV2_x2_0', 'MobileNetV2'
]


class MobileNetV2():
    def __init__(self, scale=1.0):
        self.scale = scale

    def net(self, input, class_dim=1000):
        scale = self.scale
        bottleneck_params_list = [
            (2, 64, 5, 2),
            (2, 128, 1, 2),
            (4, 128, 6, 1),
            (2, 16, 1, 1),
        ]

        conv1 = self.conv_bn_layer(
                                    input,
                                    num_filters=int(64 * scale),
                                    filter_size=3,
                                    stride=2,
                                    padding=1,
                                    if_act=True,
                                    name='conv1_01')
            
        print("conv1.shape",conv1.shape)
        
        conv2 = self.dw_conv(conv1,64,64,"dw")
        print("conv2.shape",conv2.shape)

        features = {}
        # bottleneck sequences
        i = 2
        in_c = int(64 * scale)
        input = conv2
        for layer_setting in bottleneck_params_list:
            t, c, n, s = layer_setting
            i += 1
            input = self.invresi_blocks(
                input=input,
                in_c=in_c,
                t=t,
                c=int(c * scale),
                n=n,
                s=s,
                name='conv' + str(i))
            in_c = int(c * scale)
            features['conv' + str(i)] = input
            print('conv' + str(i)+".shape",input.shape)

        x1 = features['conv6']
        print("x1.shape",x1.shape)
        
        x2 = self.conv_bn_layer(
                    input=x1,
                    num_filters=int(32 * scale),
                    filter_size=3,
                    stride=2,
                    padding=1,
                    if_act=True,
                    name='x2')
        features['x2'] = x2
        print("x2.shape",x2.shape)

        x3 = self.conv_bn_layer(
                    input=x2,
                    num_filters=int(128 * scale),
                    filter_size=7,
                    stride=1,
                    padding=0,
                    if_act=True,
                    name='x3')
        features['x3'] = x3
        print("x3.shape",x3.shape)
        
        s1 = self.conv_bn_layer(
                    input=x1,
                    num_filters=int(64 * scale),
                    filter_size=14,
                    stride=1,
                    padding=0,
                    if_act=True,
                    name='s1')
        features['s1'] = s1
        print("s1.shape",s1.shape)        
 
        s2 = self.conv_bn_layer(
                    input=x2,
                    num_filters=int(64 * scale),
                    filter_size=7,
                    stride=1,
                    padding=0,
                    if_act=True,
                    name='s2')
        features['s2'] = s2
        print("s2.shape",s2.shape) 
        s3 = x3
        print("s3.shape",s3.shape) 
        
        out = features['conv3']
        '''
        s1 = fluid.layers.pool2d(input=x1, pool_size=14, pool_type='avg', global_pooling=True, name='s1')
        s2 = fluid.layers.pool2d(input=x2, pool_size= 7, pool_type='avg', global_pooling=True, name='s2')
        s3 = x3
        print("s1.shape",s1.shape)   
        print("s2.shape",s2.shape)   
        print("s3.shape",s3.shape)   
        
        s1 = fluid.layers.flatten(s1)
        s2 = fluid.layers.flatten(s2)
        s3 = fluid.layers.flatten(s3)
        
        print("s1.shape",s1.shape)   
        print("s2.shape",s2.shape)   
        print("s3.shape",s3.shape)    
        '''

        multi_scale = fluid.layers.concat(input=[s1,s2,s3],axis=1)
        print("multi_scale.shape",multi_scale.shape)

        landmarks = fluid.layers.fc(
            input=multi_scale,
            act=None,
            size=196)
        print("landmarks.shape",landmarks.shape)
        
        angle = self.auxiliary_net(out)

        return landmarks, angle

    def conv_bn_layer(self,
                      input,
                      filter_size,
                      num_filters,
                      stride,
                      padding,
                      channels=None,
                      num_groups=1,
                      if_act=True,
                      name=None,
                      use_cudnn=True):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(name=name + '_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(
            input=conv,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')
        if if_act:
            return fluid.layers.relu6(bn)
        else:
            return bn

    def dw_conv(self,input,inp, oup, name=None):
        
        conv_dw_1 = self.conv_bn_layer(
            input=input,
            num_filters=inp,
            filter_size=3,
            stride=1,
            padding=1,
            num_groups=inp,
            if_act=False,
            use_cudnn=False,
            name='dw3' + name)

        conv_linear_1 = self.conv_bn_layer(
            input=conv_dw_1,
            num_filters=oup,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name='dw1' + name)
            
        return conv_linear_1

    def shortcut(self, input, data_residual):
        return fluid.layers.elementwise_add(input, data_residual)

    def inverted_residual_unit(self,
                               input,
                               num_in_filter,
                               num_filters,
                               ifshortcut,
                               stride,
                               filter_size,
                               padding,
                               expansion_factor,
                               name=None):
        num_expfilter = int(round(num_in_filter * expansion_factor))

        channel_expand = self.conv_bn_layer(
            input=input,
            num_filters=num_expfilter,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            name=name + '_expand')

        bottleneck_conv = self.conv_bn_layer(
            input=channel_expand,
            num_filters=num_expfilter,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            num_groups=num_expfilter,
            if_act=True,
            name=name + '_dwise',
            use_cudnn=False)

        linear_out = self.conv_bn_layer(
            input=bottleneck_conv,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=False,
            name=name + '_linear')
        if ifshortcut:
            out = self.shortcut(input=input, data_residual=linear_out)
            return out
        else:
            return linear_out

    def invresi_blocks(self, input, in_c, t, c, n, s, name=None):
        first_block = self.inverted_residual_unit(
            input=input,
            num_in_filter=in_c,
            num_filters=c,
            ifshortcut=False,
            stride=s,
            filter_size=3,
            padding=1,
            expansion_factor=t,
            name=name + '_1')
        last_residual_block = first_block
        last_c = c

        for i in range(1, n):
            last_residual_block = self.inverted_residual_unit(
                input=last_residual_block,
                num_in_filter=last_c,
                num_filters=c,
                ifshortcut=True,
                stride=1,
                filter_size=3,
                padding=1,
                expansion_factor=t,
                name=name + '_' + str(i + 1))
        return last_residual_block
        
    def auxiliary_net(self,input):
        
        conv1 = self.conv_bn_layer(
            input=input,
            num_filters=128,
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            name='aux_conv1')
        print("aux_conv1.shape",conv1.shape)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=128,
            filter_size=3,
            stride=1,
            padding=1,
            if_act=True,
            name='aux_conv2')
        print("aux_conv2.shape",conv2.shape)
        conv3 = self.conv_bn_layer(
            input=conv2,
            num_filters=32,
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            name='aux_conv3')
        print("aux_conv3.shape",conv3.shape)
        conv4 = self.conv_bn_layer(
            input=conv3,
            num_filters=128,
            filter_size=7,
            stride=1,
            padding=1,
            if_act=True,
            name='aux_conv4')
        print("aux_conv4.shape",conv4.shape)
            
        max_pool1 = fluid.layers.pool2d(
            input=conv4, pool_type='max', global_pooling=True)
        print("max_pool1.shape",max_pool1.shape)
        
        fc1 = fluid.layers.fc(input=max_pool1,size=32)
        print("fc1.shape",fc1.shape)
        fc2 = fluid.layers.fc(input=fc1,size=3)
        print("fc2.shape",fc2.shape)
        
        return fc2
        

def MobileNetV2_x0_25():
    model = MobileNetV2(scale=0.25)
    return model


def MobileNetV2_x0_5():
    model = MobileNetV2(scale=0.5)
    return model


def MobileNetV2_x0_75():
    model = MobileNetV2(scale=0.75)
    return model


def MobileNetV2_x1_0():
    model = MobileNetV2(scale=1.0)
    return  model

def build_model(img):
	
	landmarks_pre, angle = MobileNetV2(scale=1.0).net(img)
	
	
	
	return landmarks_pre,angle

def MobileNetV2_x1_5():
    model = MobileNetV2(scale=1.5)
    return model


def MobileNetV2_x2_0():
    model = MobileNetV2(scale=2.0)
    return model