import os  
import numpy as np  
import time  
import math  
import paddle  
import paddle.fluid as fluid  
  
from paddle.fluid.initializer import MSRA  
from paddle.fluid.initializer import Uniform  
from paddle.fluid.param_attr import ParamAttr  

print("model init")

score = 0.2*100 - (5.1249*np.log(160.1305)-14.499)
print(score)

class MobileNetV3():
    def __init__(self, points_num=98,model_mode="SMALL"):
        self.num_classes = int(points_num * 2)

    def h_sigmoid(self,x):
        out = fluid.layers.relu6(x+3.0, threshold=6.0)/6.0
        return out
        
    def h_swish(self,x):
        #out = x * fluid.layers.relu6(x+3.0, threshold=6.0)/6.0 
        out = fluid.layers.swish(x, beta=2.0)
        return out
        
    def activation(self,x,act):
        if act == 'HS':
            out = self.h_swish(x)
        elif act == 'RE':
            out = fluid.layers.relu(x)
        else:
            out = x
        return out
            
    def squeeze_excitation(self, input, num_channels, reduction_ratio=4, name=None):
        pool = fluid.layers.pool2d(input=input, pool_size=0, pool_type='avg', global_pooling=True) # ??
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                  size=num_channels // reduction_ratio,
                                  act='relu')
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='hard_sigmoid')
        #print("se:",input.shape,excitation.shape)
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale 
    def upsample(self,input,out_shape=None,scale=2, name=None):
        # get dynamic upsample output shape
        #shape_nchw = fluid.layers.shape(input)
        #shape_hw = fluid.layers.slice(
        #    shape_nchw, axes=[0], starts=[2], ends=[4])
        #shape_hw.stop_gradient = True
        #in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        if out_shape == None:
            print('None')
            out_shape = in_shape * scale
            out_shape.stop_gradient = True
    
        # reisze by actual_shape
        out = fluid.layers.resize_bilinear(
            input=input,out_shape=out_shape, scale=scale, name=name)#  actual_shape=out_shape,
        return out    
    def bottleneck_block(self, input, kernel_size, in_size, expand_size, out_size, semodule, nonLinear, stride,name='btb', expansion_factor=1):
        use_res_connect = stride == 1 and in_size == out_size
        expand_size = int(round(expand_size * expansion_factor))
        #  pointwise
        conv1 = fluid.layers.conv2d(input=input,
                                    num_filters=expand_size,
                                    filter_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    act=None,
                                    bias_attr=False)
        bn1 =     fluid.layers.batch_norm(input=conv1)
        
        pointwise_conv = self.activation(bn1,nonLinear)
        
        #  depthwise    
        conv2 = fluid.layers.conv2d(input=pointwise_conv,
                                    num_filters=expand_size,
                                    filter_size=kernel_size,
                                    stride=stride,
                                    padding=(kernel_size - 1) // 2,
                                    groups=expand_size,
                                    act=None,
                                    bias_attr=False)
    
        bn2 =     fluid.layers.batch_norm(input=conv2)
                                        
        
        if semodule:
            bn2 = self.squeeze_excitation(bn2, expand_size)
        
        depthwise_conv  = self.activation(bn2,nonLinear)
        
        # FC
        conv3 = fluid.layers.conv2d(input=depthwise_conv,
                                    num_filters=out_size,
                                    filter_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1,
                                    act=None,
                                    bias_attr=False)
    
        out =     fluid.layers.batch_norm(input=conv3)
                                        
        # shortcut
        if use_res_connect:
            return fluid.layers.elementwise_add(x=input, y=out)
        else:
            return out    
            
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
        
    def large_net(self,img,name="large_net"):
        print("self.img.shape",img.shape)
        # 112*112
        conv1 = fluid.layers.conv2d(    img,
                                        num_filters=16,
                                        filter_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=1,
                                        act=None,
                                        bias_attr=False)
        print("conv1.shape",conv1.shape)
        bn_name = name + "_bn"
        bn1 =     fluid.layers.batch_norm(input=conv1,act=None)
        print("bn1.shape",bn1.shape)
        hs1 = self.activation(bn1,"HS")
        print("hs1.shape",hs1.shape)
        block = hs1
        in_size = 16
        features = {}
        block = self.bottleneck_block(block, 3, 16, 16, 16, False, 'RE', 1, name=None)
        print("block.shape",block.shape)
        # 56*56
        block = self.bottleneck_block(block, 3, 16, 64, 24, False, 'RE', 2, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 24, 72, 24, False, 'RE', 1, name=None)
        print("block.shape",block.shape)

        features['auxiliary_input'] = block        

        # 28*28
        block = self.bottleneck_block(block, 5, 24, 72, 40, True, 'RE', 2, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 5, 40, 120, 40, True, 'RE', 1, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 5, 40, 120, 40, True, 'RE', 1, name=None)
        print("block.shape",block.shape)
        
        # 14*14
        block = self.bottleneck_block(block, 3, 40, 240, 80, False, 'HS', 2, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 80, 200, 80, False, 'HS', 1, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 80, 184, 80, False, 'HS', 1, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 5, 80, 184, 80, False, 'HS', 1, name=None)
        print("block.shape",block.shape)
        module14a = block
        block = self.bottleneck_block(block, 3, 80, 480, 112, True, 'HS', 1, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 112, 672, 112, True, 'HS', 1, name=None)
        print("block.shape",block.shape)
        module14b = block
        # 7*7
        block = self.bottleneck_block(block, 3, 112, 672, 160, True, 'HS', 2, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 160, 672, 160, True, 'HS', 1, name=None)
        print("block.shape",block.shape)
        block = self.bottleneck_block(block, 3, 160, 960, 160, True, 'HS', 1, name=None)
        print("block.shape",block.shape)
        module7 = block
        block = self.bottleneck_block(block, 7, 160, 960, 160, True, 'HS', 1, name=None)
        print("block.shape",block.shape)
        
        

        print("module14a.shape",module14a.shape)
        print("module14b.shape",module14b.shape)
        print("module7.shape",module7.shape)


        s1 = fluid.layers.flatten(module14a)
        s2 = fluid.layers.flatten(module14b)
        s3 = fluid.layers.flatten(module7)
        
        print("s1.shape",s1.shape)
        print("s2.shape",s2.shape)
        print("s3.shape",s3.shape)
        
        multi_scale = fluid.layers.concat(input=[s1,s2,s3],axis=1)
        print("multi_scale.shape",multi_scale.shape)

        landmarks = fluid.layers.fc(
            input=multi_scale,
            size=self.num_classes)        

        print("landmarks.shape",landmarks.shape)
        
        angle = self.auxiliary_net(features['auxiliary_input'])

        return landmarks, angle
        

def build_model(img):
    
    model = MobileNetV3()
    
    landmarks_pre, angle = model.large_net(img)
    
    return landmarks_pre,angle
