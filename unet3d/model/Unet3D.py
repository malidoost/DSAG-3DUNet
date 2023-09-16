from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Input, Conv3D, Concatenate, Dropout
from keras.layers import BatchNormalization, Activation, Add
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam, SGD
import keras.backend as K
import functools
from keras.layers import Input, Conv3D, Concatenate, Dropout
from keras.layers import BatchNormalization, Activation, Add
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam
from ..metrics import weighted_dice_coefficient_loss
from ..metrics import focal_tversky
from ..metrics import tversky_loss
from ..metrics import dsc
import tensorflow as tf
from keras.layers import *
from keras import activations
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.engine.topology import get_source_inputs
from keras.layers import Activation
from keras.layers import AveragePooling3D
from keras.layers import BatchNormalization
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling3D
from keras.layers import GlobalMaxPooling3D
from keras.layers import Input
from keras.layers import MaxPooling3D
from keras.layers import Reshape
from keras.layers import UpSampling3D
from keras.layers import concatenate
from keras.models import Model
from keras.regularizers import l2
from keras_contrib.layers import SubPixelUpscaling
from keras.layers import Activation, add, multiply, Lambda
import numpy as np
import tensorflow as tf 
#import losses 
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc, precision_recall_curve # roc curve tools
from sklearn.model_selection import train_test_split

K.set_image_data_format('channels_first')  # TF dimension ordering in this code
kinit = 'glorot_normal'
net_optimizer = Adam(1e-4)


def UnetConv3D(input, outdim, is_batchnorm, name):
	x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name+'_1')(input)
	if is_batchnorm:
		x =BatchNormalization(name=name + '_1_bn')(x)
	x = Activation('relu',name=name + '_1_act')(x)

	x = Conv3D(outdim, (3, 3, 3), strides=(1, 1, 1), kernel_initializer=kinit, padding="same", name=name+'_2')(x)
	if is_batchnorm:
		x = BatchNormalization(name=name + '_2_bn')(x)
	x = Activation('relu', name=name + '_2_act')(x)
	return x


def unet3D(opt,input_size, lossfxn, is_batchnorm):   
	
	axis = 1 if K.image_data_format()=="channels_first" else -1

	inputs = Input(shape=input_size)
	conv1 = UnetConv3D(inputs, 16, is_batchnorm=is_batchnorm, name='conv1')
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	
	conv2 = UnetConv3D(pool1, 32, is_batchnorm=is_batchnorm, name='conv2')
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = UnetConv3D(pool2, 64, is_batchnorm=is_batchnorm, name='conv3')
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = UnetConv3D(pool3, 128, is_batchnorm=is_batchnorm, name='conv4')
	pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

	conv5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(pool4)
	conv5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv5)
	
	up6 = concatenate([Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv5), conv4], axis=axis)
	conv6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up6)
	conv6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv6)

	up7 = concatenate([Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv6), conv3], axis=axis)
	conv7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up7)
	conv7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv7)

	up8 = concatenate([Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv7), conv2], axis=axis)
	conv8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(up8)

	up9 = concatenate([Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), kernel_initializer=kinit, padding='same')(conv8), conv1], axis=axis)
	conv9 = Conv3D(16, (3, 3, 3), activation='relu',  kernel_initializer=kinit, padding='same')(up9)
	conv9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kinit, padding='same')(conv9)
	conv10 = Conv3D(1, (1, 1,1), activation='sigmoid', name='final')(conv9)

	model = Model(inputs=[inputs], outputs=[conv10])
	model.compile(optimizer=opt, loss=lossfxn, metrics=[losses.dsc,losses.tp,losses.tn])
	return model


def expend_as(tensor, rep,name):
	axis = 1 if K.image_data_format()=="channels_first" else -1
	my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=axis), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
	return my_repeat


def AttnGatingBlock(x, g, inter_shape, name):
	''' take g which is the spatially smaller signal, do a conv to get the same
	number of feature channels as x (bigger spatially)
	do a conv on x to also get same geature channels (theta_x)
	then, upsample g to be same size as x 
	add x and g (concat_xg)
	relu, 1x1 conv, then sigmoid then upsample the final - this gives us attn coefficients'''
	axis = 1 if K.image_data_format()=="channels_first" else -1
	a_adapter = 1 if K.image_data_format()=="channels_first" else 0 
	shape_x = K.int_shape(x)  # 32
	shape_g = K.int_shape(g)  # 16

	theta_x = Conv3D(inter_shape, (2, 2, 2), strides=(2, 2, 2), padding='same', name='xl'+name)(x)  # 16
	shape_theta_x = K.int_shape(theta_x)

	phi_g = Conv3D(inter_shape, (1, 1, 1), padding='same')(g)
	upsample_g = Conv3DTranspose(inter_shape, (3, 3, 3),strides=(shape_theta_x[1+a_adapter] // shape_g[1+a_adapter], shape_theta_x[2+a_adapter] // shape_g[2+a_adapter],
		shape_theta_x[3+a_adapter] // shape_g[3+a_adapter]),padding='same', name='g_up'+name)(phi_g)  # 16

	concat_xg = add([upsample_g, theta_x])
	act_xg = Activation('relu')(concat_xg)
	psi = Conv3D(1, (1, 1, 1), padding='same', name='psi'+name)(act_xg)
	sigmoid_xg = Activation('sigmoid')(psi)
	shape_sigmoid = K.int_shape(sigmoid_xg)
	upsample_psi = UpSampling3D(size=(shape_x[1+a_adapter] // shape_sigmoid[1+a_adapter], shape_x[2+a_adapter] // shape_sigmoid[2+a_adapter],
	shape_x[3+a_adapter] // shape_sigmoid[3+a_adapter]))(sigmoid_xg)  # 32

	upsample_psi = expend_as(upsample_psi, shape_x[axis],  name)
	y = multiply([upsample_psi, x], name='q_attn'+name)

	result = Conv3D(shape_x[axis], (1, 1, 1), padding='same',name='q_attn_conv'+name)(y)
	result_bn = BatchNormalization(name='q_attn_bn'+name)(result)
	return result_bn


def UnetGatingSignal(input, is_batchnorm, name):
	''' this is simply 1x1 convolution, bn, activation '''
	axis = 1 if K.image_data_format()=="channels_first" else -1
	shape = K.int_shape(input)
	x = Conv3D(shape[axis] * 1, (1, 1, 1), strides=(1, 1, 1), padding="same",  kernel_initializer=kinit, name=name + '_conv')(input)
	if is_batchnorm:
		x = BatchNormalization(name=name + '_bn')(x)
	x = Activation('relu', name = name + '_act')(x)
	return x

# plain old attention gates in u-net, NO multi-input, NO deep supervision
def attn_unet_3D(opt,input_size, lossfxn, is_batchnorm): 
	axis = 1 if K.image_data_format()=="channels_first" else -1  
	inputs = Input(shape=input_size)
	conv1 = UnetConv3D(inputs, 16, is_batchnorm=is_batchnorm, name='conv1')
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	
	conv2 = UnetConv3D(pool1, 16, is_batchnorm=is_batchnorm, name='conv2')
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = UnetConv3D(pool2, 32, is_batchnorm=is_batchnorm, name='conv3')
	#conv3 = Dropout(0.2,name='drop_conv3')(conv3)
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

	conv4 = UnetConv3D(pool3, 32, is_batchnorm=is_batchnorm, name='conv4')
	#conv4 = Dropout(0.2, name='drop_conv4')(conv4)
	pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
	
	center = UnetConv3D(pool4, 64, is_batchnorm=is_batchnorm, name='center')
	
	g1 = UnetGatingSignal(center, is_batchnorm=is_batchnorm, name='g1')
	attn1 = AttnGatingBlock(conv4, g1, 64, '_1')
	up1 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], axis = axis, name='up1')
	
	g2 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g2')
	attn2 = AttnGatingBlock(conv3, g2, 32, '_2')
	up2 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], axis = axis, name='up2')

	g3 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g3')
	attn3 = AttnGatingBlock(conv2, g3, 16, '_3')
	up3 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], axis = axis, name='up3')

	up4 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], axis = axis,  name='up4')
	out = Conv3D(1, (1, 1, 1), activation='sigmoid',  kernel_initializer=kinit, name='final')(up4)
	
	model = Model(inputs=[inputs], outputs=[out])
	model.compile(optimizer=opt, loss=lossfxn, metrics=[losses.dsc,losses.tp,losses.tn])
	return model



def attn_reg_ds_3D(opt,input_size, lossfxn, is_batchnorm):
	axis = 1 if K.image_data_format()=="channels_first" else -1  
  
	img_input = Input(shape=input_size, name='input_scale1')

	conv1 = UnetConv3D(img_input, 16, is_batchnorm=is_batchnorm, name='conv1')
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	
	conv2 = UnetConv3D(pool1, 32, is_batchnorm=is_batchnorm, name='conv2')
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

	conv3 = UnetConv3D(pool2, 64, is_batchnorm=is_batchnorm, name='conv3')
	#conv3 = Dropout(0.2,name='drop_conv3')(conv3)
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
	
	conv4 = UnetConv3D(pool3, 32, is_batchnorm=is_batchnorm, name='conv4')
	#conv4 = Dropout(0.2, name='drop_conv4')(conv4)
	pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
		
	center = UnetConv3D(pool4, 256, is_batchnorm=is_batchnorm, name='center')
	
	g1 = UnetGatingSignal(center, is_batchnorm=is_batchnorm, name='g1')
	attn1 = AttnGatingBlock(conv4, g1, 64, '_1')
	up1 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], axis = axis, name='up1')

	g2 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g2')
	attn2 = AttnGatingBlock(conv3, g2, 32, '_2')
	up2 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], axis = axis, name='up2')

	g3 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g3')
	attn3 = AttnGatingBlock(conv2, g3, 16, '_3')
	up3 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], axis = axis, name='up3')

	up4 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], axis = axis, name='up4')
	
	conv6 = UnetConv3D(up1, 128, is_batchnorm=is_batchnorm, name='conv6')
	conv7 = UnetConv3D(up2, 64, is_batchnorm=is_batchnorm, name='conv7')
	conv8 = UnetConv3D(up3, 32, is_batchnorm=is_batchnorm, name='conv8')
	conv9 = UnetConv3D(up4, 16, is_batchnorm=is_batchnorm, name='conv9')

	out6 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred1')(conv6)
	out7 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred2')(conv7)
	out8 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred3')(conv8)
	out9 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='final')(conv9)

	model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
 
	loss = {'pred1':lossfxn,
			'pred2':lossfxn,
			'pred3':lossfxn,
			'final':tversky_loss}
	
	loss_weights = {'pred1':1,
					'pred2':1,
					'pred3':1,
					'final':1}
	model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
				  metrics=[losses.dsc])
	return model


#model proposed in my paper - improved attention u-net with multi-scale input pyramid and deep supervision

def attn_reg_3D(opt=SGD,initial_learning_rate=1e-2,input_size=(1,128,128,128), lossfxn=focal_tversky, n_labels=1, is_batchnorm=True):
	axis = 1 if K.image_data_format()=="channels_first" else -1  
	
	img_input = Input(shape=input_size, name='input_scale1')
	scale_img_2 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale2')(img_input)
	scale_img_3 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale3')(scale_img_2)
	scale_img_4 = AveragePooling3D(pool_size=(2, 2, 2), name='input_scale4')(scale_img_3)

	conv1 = UnetConv3D(img_input, 16, is_batchnorm=is_batchnorm, name='conv1')
	pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	
	input2 = Conv3D(32, (3, 3, 3), padding='same', activation='relu', name='conv_scale2')(scale_img_2)
	input2 = concatenate([input2, pool1], axis=axis)
	conv2 = UnetConv3D(input2, 32, is_batchnorm=is_batchnorm, name='conv2')
	pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
	
	input3 = Conv3D(64, (3, 3, 3), padding='same', activation='relu', name='conv_scale3')(scale_img_3)
	input3 = concatenate([input3, pool2], axis=axis)
	conv3 = UnetConv3D(input3, 64, is_batchnorm=is_batchnorm, name='conv3')
	pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
	
	input4 = Conv3D(128, (3, 3, 3), padding='same', activation='relu', name='conv_scale4')(scale_img_4)
	input4 = concatenate([input4, pool3], axis=axis)
	conv4 = UnetConv3D(input4, 32, is_batchnorm=is_batchnorm, name='conv4')
	pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)
		
	center = UnetConv3D(pool4, 256, is_batchnorm=is_batchnorm, name='center')
	
	g1 = UnetGatingSignal(center, is_batchnorm=is_batchnorm, name='g1')
	attn1 = AttnGatingBlock(conv4, g1, 64, '_1')
	up1 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(center), attn1], axis = axis, name='up1')

	g2 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g2')
	attn2 = AttnGatingBlock(conv3, g2, 32, '_2')
	up2 = concatenate([Conv3DTranspose(32, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up1), attn2], axis = axis, name='up2')

	g3 = UnetGatingSignal(up1, is_batchnorm=is_batchnorm, name='g3')
	attn3 = AttnGatingBlock(conv2, g3, 16, '_3')
	up3 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up2), attn3], axis = axis, name='up3')

	up4 = concatenate([Conv3DTranspose(16, (3,3,3), strides=(2,2,2), padding='same', activation='relu', kernel_initializer=kinit)(up3), conv1], axis = axis, name='up4')
	
	conv6 = UnetConv3D(up1, 128, is_batchnorm=is_batchnorm, name='conv6')
	conv7 = UnetConv3D(up2, 64, is_batchnorm=is_batchnorm, name='conv7')
	conv8 = UnetConv3D(up3, 32, is_batchnorm=is_batchnorm, name='conv8')
	conv9 = UnetConv3D(up4, 16, is_batchnorm=is_batchnorm, name='conv9')

	out6 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred1')(conv6)
	out7 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred2')(conv7)
	out8 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='pred3')(conv8)
	out9 = Conv3D(1, (1, 1, 1), activation='sigmoid', name='final')(conv9)

	model = Model(inputs=[img_input], outputs=[out6, out7, out8, out9])
 
	loss = {'pred1':lossfxn,
			'pred2':lossfxn,
			'pred3':lossfxn,
			'final':tversky_loss}
	
	loss_weights = {'pred1':1,
					'pred2':1,
					'pred3':1,
					'final':1}
	model.compile(optimizer=opt(lr=initial_learning_rate), loss=loss, loss_weights=loss_weights, metrics=[dsc])
	#model.compile(optimizer=opt, loss=loss, loss_weights=loss_weights,
				  #metrics=[losses.dsc])
	#model.compile(optimizer=Adam(1e-4), loss='mae', loss_weights=loss_weights)
	return model



#if __name__ == '__main__':
		#inp = Input((128,128,1))
	#model = unet3D(opt=Adam(1e-4), input_size=(1,64,128,128), lossfxn ='mae', is_batchnorm=False)
	#model = attn_reg_3D(opt=Adam(1e-4), input_size=(1,64,128,128), lossfxn ='mae', is_batchnorm=False)
	#print(model.summary())
	#for layer in model.layers:
		#print(layer.output_shape)


