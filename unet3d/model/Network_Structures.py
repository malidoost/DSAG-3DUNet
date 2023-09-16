from keras.layers import Input, Conv3D, Concatenate, Dropout
from keras.layers import BatchNormalization, Activation, Add
from keras.models import Model
from keras_contrib.layers import InstanceNormalization
from keras.optimizers import Adam

from ..metrics import weighted_dice_coefficient_loss


# This code contains 3D dilated-cross hair network: 
# This file will be updated in comming weeks to include more structures. 

def CrossHair_block(x_in, nb_out_channel, k_size, d_rate , act_func, d_format = "channels_first", using_act_func = True,  padding ='same', batchnorm = False, dropout = 0.0, InstanceNorm = False):
	
	"""k_size should be a one single number and stands for kernel_size. 
	x_in is the input tensor, nb_out_channel is the number of channels that we expect to have after addition.
	batchnorm and InstanceNormalization by default were set to false! feel free to turn it on:)"""
	
	x1 = Conv3D(filters = nb_out_channel, kernel_size=(k_size,k_size,1), padding=padding, 
				dilation_rate=d_rate, data_format = d_format)(x_in)
	x1 = BatchNormalization()(x1) if batchnorm else x1
	x1 = InstanceNormalization()(x1) if InstanceNorm else x1
	x1 = Activation(act_func)(x1) if using_act_func else x1
	x1 = Dropout(dropout)(x1) if dropout > 0 else x1
	
	x2 = Conv3D(filters = nb_out_channel, kernel_size=(k_size, 1 ,k_size), padding=padding,
				dilation_rate=d_rate, data_format = d_format)(x_in)
	x2 = BatchNormalization()(x2) if batchnorm else x2
	x2 = InstanceNormalization()(x2) if InstanceNorm else x2
	x2 = Activation(act_func)(x2) if using_act_func else x2
	x2 = Dropout(dropout)(x2) if dropout > 0 else x2
	
	x3 = Conv3D(filters = nb_out_channel, kernel_size=(1, k_size, k_size), padding=padding,
				dilation_rate=d_rate, data_format = d_format)(x_in)
	x3 = BatchNormalization()(x3) if batchnorm else x3
	x3 = InstanceNormalization()(x3) if InstanceNorm else x3
	x3 = Activation(act_func)(x3) if using_act_func else x3
	x3 = Dropout(dropout)(x3) if dropout > 0 else x3
	
	x_out = Add()([x1, x2, x3])
	
	return x_out



def Residual_CrossHair_block(x_in, num_filters, kernel_size, activation_func, channel_order = "channels_first", padding ='same',InstanceNorm = False, batchnorm = True):
	
	x1 = CrossHair_block(x_in, nb_out_channel = num_filters, k_size = kernel_size, act_func = activation_func, d_format = channel_order, d_rate=1, using_act_func = True, 
						 padding ='same', batchnorm = batchnorm, dropout = 0.0, InstanceNorm = InstanceNorm)
	x1 = CrossHair_block(x1, nb_out_channel = num_filters, k_size = kernel_size, act_func = activation_func, d_format = channel_order, d_rate= 1, using_act_func = False,
						 padding ='same', batchnorm = False, dropout = 0.0, InstanceNorm = False)
	x_out = Add()([x_in, x1])
	
	return x_out



#def Dilated_CrossHair_model_FixedFilterNumber(input_shape=(1, 128, 128, 128), kernel_size = 3, dilation_rate =1,  n_filters=16, activation_func = 'relu', depth=5, dropout_rate=0.3,
#					  channel_ordering = "channels_first", using_batchnorm = False,  n_labels=1, 
#					   dropoutrate = 0.0, using_instancenorm = False, initial_learning_rate=5e-4, optimizer=Adam,  loss_function=focal_loss(),
#											  last_activation="sigmoid"):
#	x_in = Input(input_shape)
#	x_out1 = CrossHair_block(x_in, nb_out_channel = n_filters, k_size = kernel_size, d_rate = dilation_rate , act_func = activation_func, 
#							 d_format = channel_ordering, using_act_func = True, padding ='same', batchnorm = using_batchnorm, dropout = dropoutrate ,
#							 InstanceNorm = using_instancenorm)
#	for nb_block in range(depth-1):
#		x_out1 = CrossHair_block(x_out1, nb_out_channel = n_filters, k_size = kernel_size, d_rate = dilation_rate , act_func = activation_func, 
#							 d_format = channel_ordering, using_act_func = True, padding ='same', batchnorm = using_batchnorm, dropout = dropoutrate ,
#							 InstanceNorm = using_instancenorm)
#	logits_output = Conv3D(filters = n_labels, kernel_size=(1,1,1), padding='same', dilation_rate=1, data_format = channel_ordering)(x_out1)
#	sig_output = Activation(last_activation)(logits_output)
#	model = Model(x_in, sig_output)
#	model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
#	return model 


#def Residual_CrossHairNet(input_shape=(1, 128, 128, 128), kernel_size = 3, n_filters=16, activation_func = 'relu', depth=5,channel_ordering = "channels_first", n_labels=1, using_instancenorm = False,
#						 initial_learning_rate=5e-4, optimizer=Adam,  loss_function=focal_loss(), last_activation="sigmoid"):
#	
#	x_in = Input(input_shape)
#	x_out1 = CrossHair_block(x_in, nb_out_channel = n_filters, k_size = kernel_size, d_rate = 1 , act_func = activation_func, d_format = channel_ordering, padding ='same', batchnorm = True, 
#							 dropout = 0.0, InstanceNorm = using_instancenorm)
#	for nb_block in range(depth):
#		x_out1 = Residual_CrossHair_block(x_out1, num_filters = n_filters, kernel_size = kernel_size, activation_func = activation_func, channel_order = channel_ordering, 
#							  padding ='same',InstanceNorm = False, batchnorm=True)
#		
#	logits_output = Conv3D(filters = n_labels, kernel_size=(1,1,1), padding='same', dilation_rate=1, data_format = channel_ordering)(x_out1)
#	sig_output = Activation(last_activation)(logits_output)
#	
#	model = Model(x_in, sig_output)
#
#	return model


def CrossHair_Dilated_Densenet(input_shape=(1, 128, 128, 128), kernel_size = 3, n_labels=1, n_filters=24, depth=5, dilation_rate = 2, activation_func = 'relu',
					 padding='same', using_batchnorm = False, using_instancenorm = True, last_activation="sigmoid",initial_learning_rate=5e-4, optimizer=Adam, 
					 channel_ordering = "channels_first", dropoutrate = 0.0,  loss_function = weighted_dice_coefficient_loss):
	x = Input(input_shape)
	inputs = x
	
	# initial convolution
	x = CrossHair_block(x, nb_out_channel = n_filters, k_size = kernel_size, d_rate = dilation_rate , act_func = activation_func, 
							 d_format = channel_ordering, using_act_func=True, padding ='same', batchnorm = using_batchnorm, dropout = dropoutrate ,
							 InstanceNorm = using_instancenorm)

	maps = [inputs]
	kernel_size = kernel_size
	for n in range(depth):
		maps.append(x)
		axis = 1 if channel_ordering == "channels_first" else -1
		x = Concatenate(axis=axis)(maps)
		x = BatchNormalization()(x) if using_batchnorm else x
		x = InstanceNormalization()(x) if using_instancenorm else x
		x = Activation('relu')(x)
		x = CrossHair_block(x, nb_out_channel = n_filters, k_size = kernel_size, d_rate = dilation_rate , act_func = activation_func, 
							 d_format = channel_ordering,using_act_func=True, padding ='same', batchnorm = False, dropout = dropoutrate ,
							 InstanceNorm = False)
		dilation_rate *= 2

	logits_output = Conv3D(filters = n_labels, kernel_size=(3,3,3), padding='same', dilation_rate=1, data_format = channel_ordering, name = "logits")(x)
	sig_output = Activation(last_activation)(logits_output)
	model = Model(inputs, sig_output)
	model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
	return model








