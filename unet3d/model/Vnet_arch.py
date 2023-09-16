
import functools
from keras.layers import Input, Conv3D, Concatenate, Dropout
from keras.layers import BatchNormalization, Activation, Add
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam
from ..metrics import weighted_dice_coefficient_loss
from ..metrics import focal_loss_fixed
import tensorflow as tf
from keras.layers import *
from keras import activations
from keras.layers.advanced_activations import PReLU
from keras.models import Model
from keras.optimizers import Adam

def downward_layer(input_layer, n_convolutions, n_output_channels, d_format = 'channels_first'):
	inl = input_layer

	for _ in range(n_convolutions):
		inl = PReLU()(
			Conv3D(filters=(n_output_channels // 2), kernel_size=5,
				   padding='same', kernel_initializer='he_normal', data_format = d_format)(inl)
		)
	add_l = add([inl, input_layer])
	downsample = Conv3D(filters=n_output_channels, kernel_size=2, strides=2,
						padding='same', kernel_initializer='he_normal',data_format = d_format)(add_l)
	downsample = PReLU()(downsample)
	return downsample, add_l


def upward_layer(input0, input1, n_convolutions, n_output_channels, d_format = 'channels_first'):
	axis = 1 if d_format == 'channels_first' else -1
	merged = concatenate([input0, input1], axis=axis)
	inl = merged
	for _ in range(n_convolutions):
		inl = PReLU()(
			Conv3D((n_output_channels * 4), kernel_size=5,
				   padding='same', kernel_initializer='he_normal',data_format = d_format)(inl)
		)
	add_l = add([inl, merged])
	shape = add_l.get_shape().as_list()
	new_shape = (1, shape[1] * 2, shape[2] * 2, shape[3] * 2, n_output_channels)
	upsample = Conv3DTranspose(n_output_channels, (2,2,2), strides=(2, 2, 2),data_format = d_format)(add_l)
	#upsample = Deconvolution3D(n_output_channels, (2, 2, 2), new_shape, subsample=(2, 2, 2))(add_l)
	return PReLU()(upsample)


def vnet(input_shape=(1, 128, 128, 128), initial_learning_rate=5e-4, optimizer=Adam,
		 loss_function=weighted_dice_coefficient_loss, n_labels=1, d_format = 'channels_first'):
		 # loss='categorical_crossentropy', metrics=['categorical_accuracy']):
	# Layer 1
	inputs = Input(input_shape)
	conv1 = Conv3D(16, kernel_size=5, strides=1, padding='same', kernel_initializer='he_normal',data_format = d_format)(inputs)
	conv1 = PReLU()(conv1)
	axis = 1 if d_format == 'channels_first' else -1
	repeat1 = concatenate(16 * [inputs], axis=axis)
	add1 = add([conv1, repeat1])
	down1 = Conv3D(32, 2, strides=2, padding='same', kernel_initializer='he_normal',data_format = d_format)(add1)
	down1 = PReLU()(down1)

	# Layer 2,3,4
	down2, add2 = downward_layer(down1, 2, 64,d_format = d_format)
	down3, add3 = downward_layer(down2, 3, 128,d_format = d_format)
	down4, add4 = downward_layer(down3, 3, 256,d_format = d_format)

	# Layer 5
	# !Mudar kernel_size=(5, 5, 5) quando imagem > 64!
	conv_5_1 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',data_format = d_format)(down4)
	conv_5_1 = PReLU()(conv_5_1)
	conv_5_2 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',data_format = d_format)(conv_5_1)
	conv_5_2 = PReLU()(conv_5_2)
	conv_5_3 = Conv3D(256, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',data_format = d_format)(conv_5_2)
	conv_5_3 = PReLU()(conv_5_3)
	add5 = add([conv_5_3, down4])
	aux_shape = add5.get_shape()
	#upsample_5 = Deconvolution3D(128, (2, 2, 2), (1, aux_shape[1].value*2,aux_shape[2].value*2,
												  #aux_shape[3].value*2, 128), subsample=(2, 2, 2))(add5)
	upsample_5 = Conv3DTranspose(128,(2,2,2),strides=(2, 2, 2),data_format = d_format)(add5)
	upsample_5 = PReLU()(upsample_5)

	# Layer 6,7,8
	upsample_6 = upward_layer(upsample_5, add4, 3, 64,d_format = d_format)
	upsample_7 = upward_layer(upsample_6, add3, 3, 32,d_format = d_format)
	upsample_8 = upward_layer(upsample_7, add2, 2, 16,d_format = d_format)

	# Layer 9
	axis = 1 if d_format == 'channels_first' else -1
	merged_9 = concatenate([upsample_8, add1], axis=axis)
	conv_9_1 = Conv3D(32, kernel_size=(5, 5, 5), padding='same', kernel_initializer='he_normal',data_format = d_format)(merged_9)
	conv_9_1 = PReLU()(conv_9_1)
	add_9 = add([conv_9_1, merged_9])
	# conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal')(add_9)
	conv_9_2 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',data_format = d_format)(add_9)
	conv_9_2 = PReLU()(conv_9_2)

	# softmax = Softmax()(conv_9_2)
	sigmoid = Conv3D(filters = n_labels, kernel_size=(1, 1, 1), padding='same', kernel_initializer='he_normal',
					 activation='sigmoid',data_format = d_format)(conv_9_2)

	model = Model(inputs=inputs, outputs=sigmoid)
	# model = Model(inputs=inputs, outputs=softmax)

	model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)

	return model
