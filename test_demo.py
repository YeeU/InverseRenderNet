import os
import numpy as np
import tensorflow as tf
import cv2
from skimage import io
import argparse
from model import SfMNet, lambSH_layer, pred_illuDecomp_layer
from utils import render_sphere_nm


parser = argparse.ArgumentParser(description='InverseRenderNet')
parser.add_argument('--image', help='Path to test image')
parser.add_argument('--mask', help='Path to image mask')
parser.add_argument('--model', help='Path to trained model')


args = parser.parse_args()

img_path = args.image
mask_path = args.mask

img = io.imread(img_path)
mask = io.imread(mask_path)


dst_dir = 'test_results'
os.makedirs(dst_dir)

input_height = 200
input_width = 200
ori_height, ori_width = img.shape[:2]

if ori_height / ori_width >1:
    scale = ori_width / 200
    input_height = np.int32(scale * 200)
else:
    scale = ori_height / 200
    input_width = np.int32(scale * 200)


# compute pseudo inverse for input matrix
def pinv(A, reltol=1e-6):
	# compute SVD of input A
	s, u, v = tf.svd(A)

	# invert s and clear entries lower than reltol*s_max
	atol = tf.reduce_max(s) * reltol
	s = tf.boolean_mask(s, s>atol)
	s_inv = tf.diag(1./s)

	# compute v * s_inv * u_t as psuedo inverse
	return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))



inputs_var = tf.placeholder(tf.float32, (None, input_height, input_width, 3))
masks_var = tf.placeholder(tf.float32, (None, input_height, input_width, 1))
train_flag = tf.placeholder(tf.bool, ())
am_deconvOut, nm_deconvOut = SfMNet.SfMNet(inputs=inputs_var,is_training=train_flag, height=input_height, width=input_width, n_layers=30, n_pools=4, depth_base=32)


# separate albedo, error mask and shadow mask from deconvolutional output
albedos = am_deconvOut
nm_pred = nm_deconvOut

gamma = tf.constant(2.2)

# post-process on raw albedo and nm_pred
albedos = tf.nn.sigmoid(albedos) * masks_var + tf.constant(1e-4)

nm_pred_norm = tf.sqrt(tf.reduce_sum(nm_pred**2, axis=-1, keepdims=True)+tf.constant(1.))
nm_pred_xy = nm_pred / nm_pred_norm
nm_pred_z = tf.constant(1.) / nm_pred_norm
nm_pred_xyz = tf.concat([nm_pred_xy, nm_pred_z], axis=-1) * masks_var


# compute illumination
lighting_model = 'illu_pca'
lighting_vectors = tf.constant(np.load(os.path.join(lighting_model,'pcaVector.npy')),dtype=tf.float32)
lighting_means = tf.constant(np.load(os.path.join(lighting_model,'mean.npy')),dtype=tf.float32)	
lightings = pred_illuDecomp_layer.illuDecomp(inputs_var, albedos, nm_pred_xyz, gamma, masks_var)


lightings_pca = tf.matmul((lightings - lighting_means), pinv(lighting_vectors))
lightings = tf.matmul(lightings_pca,lighting_vectors) + lighting_means 
# reshape 27-D lightings to 9*3 lightings
lightings = tf.reshape(lightings,[tf.shape(lightings)[0],9,3])

# visualisations
shading, _ = lambSH_layer.lambSH_layer(tf.ones_like(albedos), nm_pred_xyz, lightings, 1.)
nm_sphere = tf.constant(render_sphere_nm.render_sphere_nm(100,1),dtype=tf.float32)
nm_sphere = tf.tile(nm_sphere, (tf.shape(inputs_var)[0],1,1,1))
lighting_recon, _ = lambSH_layer.lambSH_layer(tf.ones_like(nm_sphere), nm_sphere, lightings, 1.)


irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='am') + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='nm')
model_path = tf.train.get_checkpoint_state(args.model).model_checkpoint_path

total_loss = 0 
sess = tf.InteractiveSession()
saver = tf.train.Saver(irn_vars)
saver.restore(sess, model_path)


# evaluation
ori_img = img
ori_height, ori_width = ori_img.shape[:2]
img = cv2.resize(img, (input_width, input_height))
img = np.float32(img)/255.
img = img[None, :, :, :]
mask = cv2.resize(mask, (input_width, input_height), cv2.INTER_NEAREST)
mask = np.float32(mask==255)[None,:,:,None]

[albedos_val, nm_pred_val, lighting_recon_val, shading_val] = sess.run([albedos, nm_pred_xyz, lighting_recon, shading], feed_dict={train_flag:False, inputs_var:img, masks_var:mask})


# post-process results
nm_pred_val = (nm_pred_val+1.)/2.

albedos_val = cv2.resize(albedos_val[0], (ori_width, ori_height))
shading_val = cv2.resize(shading_val[0], (ori_width, ori_height))
lighting_recon_val = lighting_recon_val[0]
nm_pred_val = cv2.resize(nm_pred_val[0], (ori_width, ori_height))


albedos_val = (albedos_val-albedos_val.min()) / (albedos_val.max()-albedos_val.min())

albedos_val = np.uint8(albedos_val*255.)
shading_val = np.uint8(shading_val*255.)
lighting_recon_val = np.uint8(lighting_recon_val*255.)
nm_pred_val = np.uint8(nm_pred_val*255.)

input_path = os.path.join(dst_dir, 'img.png')
io.imsave(input_path, ori_img)
albedo_path = os.path.join(dst_dir, 'albedo.png')
io.imsave(albedo_path, albedos_val)
shading_path = os.path.join(dst_dir, 'shading.png')
io.imsave(shading_path, shading_val)
nm_pred_path = os.path.join(dst_dir, 'nm_pred.png')
io.imsave(nm_pred_path, nm_pred_val)
lighting_path = os.path.join(dst_dir, 'lighting.png')
io.imsave(lighting_path, lighting_recon_val)


