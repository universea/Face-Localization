#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#	http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
from paddle.fluid.param_attr import ParamAttr

__all__ = ['ShuffleNetV2_x0_5_swish', 'ShuffleNetV2_x1_0_swish', 'ShuffleNetV2_x1_5_swish', 'ShuffleNetV2_x2_0_swish']

train_parameters = {
	"input_size": [3, 224, 224],
	"input_mean": [0.485, 0.456, 0.406],
	"input_std": [0.229, 0.224, 0.225],
	"learning_strategy": {
		"name": "piecewise_decay",
		"batch_size": 256,
		"epochs": [30, 60, 90],
		"steps": [0.1, 0.01, 0.001, 0.0001]
	}
}


class ShuffleNetV2():
	def __init__(self, scale=1.0):
		self.params = train_parameters
		self.scale = scale
		
	def upsample(self,input,out_shape=None,scale=2, name=None):
		# get dynamic upsample output shape
		#shape_nchw = fluid.layers.shape(input)
		#shape_hw = fluid.layers.slice(
		#	shape_nchw, axes=[0], starts=[2], ends=[4])
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
		
	def net(self, input, class_dim=81, img_shape=[3, 300, 300]):
		self.num_classes = class_dim
		self.img_shape = img_shape
		scale = self.scale 
		stage_repeats = [4, 8, 4,1,1,1,1]
		
		if scale == 0.5:
			stage_out_channels = [-1, 24,  48,  96, 192, 1024]
		elif scale == 1.0:
			stage_out_channels = [-1, 24, 116, 232, 464, 82, 82, 82, 82]
		elif scale == 1.5:
			stage_out_channels = [-1, 24, 176, 352, 704, 1024]
		elif scale == 2.0:
			stage_out_channels = [-1, 24, 224, 488, 976, 2048]
		else:
			raise ValueError(
				"""{} groups is not supported for
					   1x1 Grouped Convolutions""".format(num_groups))

		#conv1
		feature_maps = []
		input_channel = stage_out_channels[1]
		conv1 = self.conv_bn_layer(input=input, filter_size=3, num_filters=input_channel, padding=1, stride=2,name='stage1_conv')	
		pool1 = fluid.layers.pool2d(input=conv1, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
		conv = pool1
		#conv.stop_gradient = True
		# bottleneck sequences
		for idxstage in range(len(stage_repeats)):
			numrepeat = stage_repeats[idxstage]
			output_channel = stage_out_channels[idxstage+2]
			for i in range(numrepeat):
				if i == 0:
					conv = self.inverted_residual_unit(input=conv, num_filters=output_channel, stride=2, 
													   benchmodel=2,name=str(idxstage+2)+'_'+str(i+1))
				else:
					conv = self.inverted_residual_unit(input=conv, num_filters=output_channel, stride=1, 
													   benchmodel=1,name=str(idxstage+2)+'_'+str(i+1))
				print(str(idxstage+2)+'_'+str(i+1)+'.shape = ',conv.shape)
				
			if idxstage >= 1:
				feature_maps.append(conv)

		print('feature_maps[0] = ',feature_maps[0].shape)
		print('feature_maps[1] = ',feature_maps[1].shape)
		print('feature_maps[2] = ',feature_maps[2].shape)
		print('feature_maps[3] = ',feature_maps[3].shape)
		print('feature_maps[4] = ',feature_maps[4].shape)
		print('feature_maps[5] = ',feature_maps[5].shape)
		#feature_maps[0].stop_gradient = True
		#feature_maps[1].stop_gradient = True
		#feature_maps[2].stop_gradient = True
		
		temp = self.upsample(feature_maps[1],out_shape=[19,19],scale=2, name=None)
		feature_maps[0] = fluid.layers.concat(input=[feature_maps[0], temp],axis=1)
		
		temp = self.upsample(feature_maps[2],out_shape=[10,10],scale=2, name=None)
		feature_maps[1] = fluid.layers.concat(input=[feature_maps[1], temp],axis=1)
		
		temp = self.upsample(feature_maps[3],out_shape=[5,5],scale=2, name=None)
		feature_maps[2] = fluid.layers.concat(input=[feature_maps[2], temp],axis=1)
		
		temp = self.upsample(feature_maps[4],out_shape=[3,3],scale=2, name=None)
		feature_maps[3] = fluid.layers.concat(input=[feature_maps[3], temp],axis=1)
		
		temp = self.upsample(feature_maps[5],out_shape=[2,2],scale=2, name=None)
		feature_maps[4] = fluid.layers.concat(input=[feature_maps[4], temp],axis=1)
		
		print('feature_maps[0] = ',feature_maps[0].shape)
		print('feature_maps[1] = ',feature_maps[1].shape)
		print('feature_maps[2] = ',feature_maps[2].shape)
		print('feature_maps[3] = ',feature_maps[3].shape)
		print('feature_maps[4] = ',feature_maps[4].shape)
		print('feature_maps[5] = ',feature_maps[5].shape)

		mbox_locs, mbox_confs, box, box_var = fluid.layers.multi_box_head(
			inputs=feature_maps,
			image=input,
			num_classes=self.num_classes,
			min_ratio=10,
			max_ratio=90,
			min_sizes=[60.0, 105.0, 150.0, 195.0, 240.0, 285.0],
			max_sizes=[[], 150.0, 195.0, 240.0, 285.0, 300.0],
			aspect_ratios=[[2.], [2., 3.], [2., 3.], [2., 3.], [2., 3.],
						   [2., 3.]],
			base_size=self.img_shape[2],
			offset=0.5,
			flip=True)
		return mbox_locs, mbox_confs, box, box_var

	
	def conv_bn_layer(self,
				  input,
				  filter_size,
				  num_filters,
				  stride,
				  padding,
				  num_groups=1,
				  use_cudnn=True,
				  if_act=True,
				  name=None):
		conv = fluid.layers.conv2d(
			input=input,
			num_filters=num_filters,
			filter_size=filter_size,
			stride=stride,
			padding=padding,
			groups=num_groups,
			act=None,
			use_cudnn=use_cudnn,
			param_attr=ParamAttr(initializer=MSRA(),name=name+'_weights'),
			bias_attr=False)
		out = int((input.shape[2] - 1)/float(stride) + 1)
		bn_name = name + '_bn'
		if if_act:
			return fluid.layers.batch_norm(input=conv, act='swish',
										   param_attr = ParamAttr(name=bn_name+"_scale"),
										   bias_attr=ParamAttr(name=bn_name+"_offset"),
										   moving_mean_name=bn_name + '_mean',
										   moving_variance_name=bn_name + '_variance')
		else:
			return fluid.layers.batch_norm(input=conv,
										   param_attr = ParamAttr(name=bn_name+"_scale"),
										   bias_attr=ParamAttr(name=bn_name+"_offset"),
										   moving_mean_name=bn_name + '_mean',
										   moving_variance_name=bn_name + '_variance')

	  
	def channel_shuffle(self, x, groups):
		batchsize, num_channels, height, width = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
		channels_per_group = num_channels // groups
	
		# reshape
		x = fluid.layers.reshape(x=x, shape=[batchsize, groups, channels_per_group, height, width])

		x = fluid.layers.transpose(x=x, perm=[0,2,1,3,4])

		# flatten
		x = fluid.layers.reshape(x=x, shape=[batchsize, num_channels, height, width])

		return x

	
	def inverted_residual_unit(self, input, num_filters, stride, benchmodel, name=None):
		assert stride in [1, 2], \
			"supported stride are {} but your stride is {}".format([1,2], stride)
			
		oup_inc = num_filters//2
		inp = input.shape[1]
		
		if benchmodel == 1:
			x1, x2 = fluid.layers.split(
				input, num_or_sections=[input.shape[1]//2, input.shape[1]//2], dim=1)			
			
			conv_pw = self.conv_bn_layer(
				input=x2, 
				num_filters=oup_inc, 
				filter_size=1, 
				stride=1,
				padding=0,
				num_groups=1,
				if_act=True,
				name='stage_'+name+'_conv1')

			conv_dw = self.conv_bn_layer(
				input=conv_pw, 
				num_filters=oup_inc, 
				filter_size=3, 
				stride=stride, 
				padding=1,
				num_groups=oup_inc, 
				if_act=False,
				use_cudnn=False,
				name='stage_'+name+'_conv2')

			conv_linear = self.conv_bn_layer(
				input=conv_dw, 
				num_filters=oup_inc, 
				filter_size=1, 
				stride=1, 
				padding=0,
				num_groups=1, 
				if_act=True,
				name='stage_'+name+'_conv3')
			
			out = fluid.layers.concat([x1, conv_linear], axis=1)

			
		else:
			#branch1
			conv_dw_1 = self.conv_bn_layer(
				input=input, 
				num_filters=inp, 
				filter_size=3, 
				stride=stride,
				padding=1,
				num_groups=inp,
				if_act=False,
				use_cudnn=False,
				name='stage_'+name+'_conv4')
			
			conv_linear_1 = self.conv_bn_layer(
				input=conv_dw_1, 
				num_filters=oup_inc, 
				filter_size=1, 
				stride=1,
				padding=0,
				num_groups=1,
				if_act=True,
				name='stage_'+name+'_conv5')
			
			#branch2
			conv_pw_2 = self.conv_bn_layer(
				input=input, 
				num_filters=oup_inc, 
				filter_size=1, 
				stride=1,
				padding=0,
				num_groups=1,
				if_act=True,
				name='stage_'+name+'_conv1')

			conv_dw_2 = self.conv_bn_layer(
				input=conv_pw_2, 
				num_filters=oup_inc, 
				filter_size=3, 
				stride=stride, 
				padding=1,
				num_groups=oup_inc, 
				if_act=False,
				use_cudnn=False,
				name='stage_'+name+'_conv2')

			conv_linear_2 = self.conv_bn_layer(
				input=conv_dw_2, 
				num_filters=oup_inc, 
				filter_size=1, 
				stride=1, 
				padding=0,
				num_groups=1, 
				if_act=True,
				name='stage_'+name+'_conv3')
			out = fluid.layers.concat([conv_linear_1, conv_linear_2], axis=1)
			
		return self.channel_shuffle(out, 2)
	
def ShuffleNetV2_x0_5_swish():
	model = ShuffleNetV2(scale=0.5)
	return model

def ShuffleNetV2_x1_0_swish():
	model = ShuffleNetV2(scale=1.0)
	return model

def ShuffleNetV2_x1_5_swish():
	model = ShuffleNetV2(scale=1.5)
	return model

def ShuffleNetV2_x2_0_swish():
	model = ShuffleNetV2(scale=2.0)
	return model
	
def build_mobilenet_ssd(img, num_classes, img_shape):
	model = ShuffleNetV2_x1_0_swish()
	return model.net(img,num_classes,img_shape)