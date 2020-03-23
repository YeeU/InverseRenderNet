import pickle as pk
import os
import numpy as np
import tensorflow as tf
import skimage.transform as imgTform
import glob
from scipy import io


def megaDepth_dataPipeline(num_subbatch_input, dir):
	# import ipdb; ipdb.set_trace()
	# locate all scenes 
	data_scenes1 = np.array(sorted(glob.glob(os.path.join(dir, '*'))))

	# scan scenes
	# sort scenes by number of training images in each
	scenes_size1 = np.array([len(os.listdir(i)) for i in data_scenes1])
	scenes_sorted1 = np.argsort(scenes_size1)

	# define scenes for training and testing
	train_scenes = data_scenes1[scenes_sorted1]


	# load data from each scene
	# locate each data minibatch in each sorted sc
	train_scenes_items = [sorted(glob.glob(os.path.join(sc, '*.pk'))) for sc in train_scenes]
	train_scenes_items = np.concatenate(train_scenes_items, axis=0)

	train_items = train_scenes_items

	### contruct training data pipeline
	# remove residual data over number of data in one epoch
	res_train_items = len(train_items) - (len(train_items) % num_subbatch_input)
	train_items = train_items[:res_train_items]
	train_data = md_construct_inputPipeline(train_items, flag_shuffle=True, batch_size=num_subbatch_input)

	# define re-initialisable iterator
	iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
	next_element = iterator.get_next()

	# define initialisation for each iterator
	trainData_init_op = iterator.make_initializer(train_data)

	return next_element, trainData_init_op, len(train_items)


def _read_pk_function(filename):
	with open(filename, 'rb') as f:
		batch_data = pk.load(f)
	input = np.float32(batch_data['input'])
	dm = batch_data['dm']
	nm = np.float32(batch_data['nm'])
	cam = np.float32(batch_data['cam'])
	scaleX= batch_data['scaleX']
	scaleY = batch_data['scaleY']
	mask = np.float32(batch_data['mask'])

	return input, dm, nm, cam, scaleX, scaleY, mask

def md_read_func(filename):

	input, dm, nm, cam, scaleX, scaleY, mask = tf.py_func(_read_pk_function, [filename], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32])	

	input = tf.data.Dataset.from_tensor_slices(input[None])
	dm = tf.data.Dataset.from_tensor_slices(dm[None])
	nm = tf.data.Dataset.from_tensor_slices(nm[None])
	cam = tf.data.Dataset.from_tensor_slices(cam[None])
	scaleX = tf.data.Dataset.from_tensor_slices(scaleX[None])
	scaleY = tf.data.Dataset.from_tensor_slices(scaleY[None])
	mask = tf.data.Dataset.from_tensor_slices(mask[None])

	return tf.data.Dataset.zip((input, dm, nm, cam, scaleX, scaleY, mask))


def md_preprocess_func(input, dm, nm, cam, scaleX, scaleY, mask):

	input = input/255.

	nm = nm/127

	return input, dm, nm, cam, scaleX, scaleY, mask


def md_construct_inputPipeline(items, batch_size, flag_shuffle=True):
	data = tf.data.Dataset.from_tensor_slices(items)
	if flag_shuffle:
		data = data.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=100000))
	else:
		data = data.repeat()
	data = data.apply(tf.contrib.data.parallel_interleave(md_read_func, cycle_length=batch_size, block_length=1, sloppy=False ))
	data = data.map(md_preprocess_func, num_parallel_calls=8 )
	data = data.batch(batch_size).prefetch(4)

	return data


