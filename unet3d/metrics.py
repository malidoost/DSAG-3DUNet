from functools import partial
import tensorflow as tf
import keras.backend as K
from keras.losses import binary_crossentropy
epsilon = 1e-5
smooth = 1


def dice_coefficient(y_true, y_pred, smooth=1.):
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dsc(y_true, y_pred):
	smooth = 1.
	y_true_f = K.flatten(y_true)
	y_pred_f = K.flatten(y_pred)
	intersection = K.sum(y_true_f * y_pred_f)
	score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
	return score

def dice_loss(y_true, y_pred):
	loss = 1 - dsc(y_true, y_pred)
	return loss

def bce_dice_loss(y_true, y_pred):
	loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
	return loss



def dice_coefficient_loss(y_true, y_pred):
	return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
	"""
	Weighted dice coefficient. Default axis assumes a "channels first" data structure
	:param smooth:
	:param y_true:
	:param y_pred:
	:param axis:
	:return:
	"""
	return K.mean(2. * (K.sum(y_true * y_pred,
							  axis=axis) + smooth/2)/(K.sum(y_true,
															axis=axis) + K.sum(y_pred,
																			   axis=axis) + smooth))

def weighted_dice_coefficient_loss(y_true, y_pred):
	return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
	return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
	f = partial(label_wise_dice_coefficient, label_index=label_index)
	f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
	return f


def confusion(y_true, y_pred):
	smooth=1
	y_pred_pos = K.clip(y_pred, 0, 1)
	y_pred_neg = 1 - y_pred_pos
	y_pos = K.clip(y_true, 0, 1)
	y_neg = 1 - y_pos
	tp = K.sum(y_pos * y_pred_pos)
	fp = K.sum(y_neg * y_pred_pos)
	fn = K.sum(y_pos * y_pred_neg) 
	prec = (tp + smooth)/(tp+fp+smooth)
	recall = (tp+smooth)/(tp+fn+smooth)
	return prec, recall

def tp(y_true, y_pred):
	smooth = 1
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pos = K.round(K.clip(y_true, 0, 1))
	tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
	return tp 

def tn(y_true, y_pred):
	smooth = 1
	y_pred_pos = K.round(K.clip(y_pred, 0, 1))
	y_pred_neg = 1 - y_pred_pos
	y_pos = K.round(K.clip(y_true, 0, 1))
	y_neg = 1 - y_pos 
	tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
	return tn 

def tversky(y_true, y_pred):
	smooth = 1
	y_true_pos = K.flatten(y_true)
	y_pred_pos = K.flatten(y_pred)
	true_pos = K.sum(y_true_pos * y_pred_pos)
	false_neg = K.sum(y_true_pos * (1-y_pred_pos))
	false_pos = K.sum((1-y_true_pos)*y_pred_pos)
	alpha = 0.7
	return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
	return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
	pt_1 = tversky(y_true, y_pred)
	gamma = 0.75
	return K.pow((1-pt_1), gamma)



















dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
