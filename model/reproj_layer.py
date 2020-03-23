# apply error mask in albedo reprojection


# no rotation involved


#### directly output flatten reprojected pixels and the reconstruction mask

# the differentiable layer performing reprojection

import tensorflow as tf
import numpy as np

# pc is n-by-3 matrix containing point could three locations
# cam is the new camera parameters, whose f and p_a have shape (batch) and c has shape (batch, 2)
# dm1 is the depth map associated with cam1 that is camera for output image, which has shape (batch, height, width)
# img2 is the input image that acts as source image for reprojection, which has shape (batch, height, width, 3)
def map_reproj(dm1,map2,cam1,cam2,scale_x1,scale_x2,scale_y1,scale_y2):
	batch_size = tf.shape(dm1)[0]

	# read camera parameters
	c1 = cam1[:,2:4]
	f1 = cam1[:,0]
	p_a1 = cam1[:,1] # ratio is width divided by height
	R1 = tf.reshape(cam1[:,4:13],[-1,3,3])
	t1 = cam1[:,13:]

	c2 = cam2[:,2:4]
	f2 = cam2[:,0]
	p_a2 = cam2[:,1]
	R2 = tf.reshape(cam2[:,4:13],[-1,3,3])
	t2 = cam2[:,13:]

	# project pixel points back to camera coords
	# u is the height and v is the width
	# u and v are scalars
	u1 = tf.shape(dm1)[1]
	v1 = tf.shape(dm1)[2]

	# convert u1 and v1 to float, convenient for computation
	u1 = tf.to_float(u1)
	v1 = tf.to_float(v1)

	### regular grid in output image
	# x increase towards right, y increase toward down
	vm,um = tf.meshgrid(tf.range(1.,v1+1.), tf.range(1.,u1+1.))


	# apply scaling factors on f
	# f1 = f1/(scale_x1+scale_y1)*2
	# f1 = tf.stack([f1, f1*p_a1],axis=-1)
	f1 = tf.stack([f1/scale_x1, f1/scale_y1*p_a1],axis=-1)

	# expand f1 (batch,2,1,1), to be consistant with dm 
	f1 = tf.expand_dims(tf.expand_dims(f1,axis=-1),axis=-1)
	# expand c1 dimension (batch,2,1,1)
	c1 = tf.expand_dims(tf.expand_dims(c1,axis=-1),axis=-1)
	# expand vm and um to have shape (1,height,width)
	vm = tf.expand_dims(vm,axis=0)	
	um = tf.expand_dims(um,axis=0)	

	# compute 3D point x and y coordinates
	# Xm and Ym have shape (batch, height, width)
	Xm = (vm-c1[:,0])/f1[:,0]*dm1
	Ym = (um-c1[:,1])/f1[:,1]*dm1

	# the point cloud is (batch, 3, npix) matrix, each row is XYZ cam coords for one point
	pc = tf.stack([tf.contrib.layers.flatten(Xm), tf.contrib.layers.flatten(Ym), tf.contrib.layers.flatten(dm1)], axis=1)

	### transfer pc from coords of cam1 to cam2
	# construct homogeneous point cloud with shape batch-4-by-num_pix
	num_pix = tf.shape(pc)[-1]
	homo_pc_c1 = tf.concat([pc, tf.ones((batch_size,1,num_pix), dtype=tf.float32)], axis=1)

	# both transformation matrix have shape batch-by-4-by-4, valid for multiplication with defined homogeneous point cloud
	last_row = tf.tile(tf.constant([[[0,0,0,1]]],dtype=tf.float32), multiples=[batch_size,1,1])
	W_C_R_t1 = tf.concat([R1,tf.expand_dims(t1,axis=2)],axis=2)
	W_C_trans1 = tf.concat([W_C_R_t1, last_row], axis=1)
	W_C_R_t2 = tf.concat([R2,tf.expand_dims(t2,axis=2)],axis=2)
	W_C_trans2 = tf.concat([W_C_R_t2, last_row], axis=1)

	# batch dot product, output has shape (batch, 4, npix)	
	homo_pc_c2 = tf.matmul(W_C_trans2, tf.matmul(tf.matrix_inverse(W_C_trans1), homo_pc_c1))

	### project point cloud to cam2 pixel coordinates
	# u in vertical and v in horizontal
	u2 = tf.shape(map2)[1]
	v2 = tf.shape(map2)[2]

	# convert u2 and v2 to float
	u2 = tf.to_float(u2)
	v2 = tf.to_float(v2)
	
	# f2 = f2/(scale_x2+scale_y2)*2
	# f2 = tf.stack([f2, f2*p_a2],axis=-1)
	f2 = tf.stack([f2/scale_x2, f2/scale_y2*p_a2],axis=-1)

	# construct intrics matrics, which has shape (batch, 3, 4)
	zeros = tf.zeros_like(f2[:,0],dtype=tf.float32)	
	ones = tf.ones_like(f2[:,0],tf.float32)
	k2 = tf.stack([tf.stack([f2[:,0],zeros,c2[:,0],zeros],axis=1), tf.stack([zeros,f2[:,1],c2[:,1],zeros],axis=1), tf.stack([zeros,zeros,ones,zeros],axis=1)],axis=1)

	## manual batch dot product
	k2 = tf.expand_dims(k2,axis=-1)
	homo_pc_c2 = tf.expand_dims(homo_pc_c2,axis=1)
	# homo_uv2 has shape (batch, 3, npix)
	homo_uv2 = tf.reduce_sum(k2*homo_pc_c2,axis=2)

	# the reprojected locations of regular grid in output image
	# both have shape (batch, npix)
	v_reproj = homo_uv2[:,0,:]/homo_uv2[:,2,:]
	u_reproj = homo_uv2[:,1,:]/homo_uv2[:,2,:]

	# u and v are flatten vector containing reprojected pixel locations
	# the u and v on same index compose one pixel 
	u_valid = tf.logical_and(tf.logical_and(tf.logical_not(tf.is_nan(u_reproj)), u_reproj>0), u_reproj<u2-1)
	v_valid = tf.logical_and(tf.logical_and(tf.logical_not(tf.is_nan(v_reproj)), v_reproj>0), v_reproj<v2-1)
	# pixels has shape (batch, npix), indicating available reprojected pixels
	pixels = tf.logical_and(u_valid,v_valid)

	# pixels is bool indicator over original regular grid
	# v_reproj and u_reproj is x and y coordinates in source image
	# pixels, v_reproj and u_reproj are corresponded with each other by their indices

	### interpolation function based on source image img2
	# it has shape (total_npix, 3), the second dimension contains [img_inds, x, y]; we need to use img_inds to distinguish each pixel's request image
	# img_inds is 2d matrix with shape (batch, npix), containing img_ind for each (x,y) location
	img_inds = tf.tile(tf.expand_dims(tf.to_float(tf.range(batch_size)), axis=1), multiples=[1,num_pix])
	request_points1 = tf.stack([tf.boolean_mask(img_inds,pixels), tf.boolean_mask(v_reproj,pixels), tf.boolean_mask(u_reproj,pixels)], axis=1)



	# the output is stacked flatten pixel values for channels
	re_proj_pixs = interpImg(request_points1, map2)	

	# reconstruct original shaped re-projection map
	ndims = tf.shape(map2)[3]
	shape = [batch_size, tf.to_int32(u1), tf.to_int32(v1),3]

	pixels = tf.reshape(pixels,shape=tf.stack([batch_size, tf.to_int32(u1), tf.to_int32(v1)],axis=0))
	indices = tf.to_int32(tf.where(tf.equal(pixels,True)))

	re_proj_pixs = tf.scatter_nd(updates=re_proj_pixs, indices=indices, shape=shape)

	# re_proj_pix is flatten reprojection results with shape (total_npix, 3)
	# indices contains first three indices in original image shape for each pixel in re_proj_pixs
	return re_proj_pixs, pixels



def interpImg(unknown,data):
	# interpolate unknown data on pixel locations defined in unknown from known data with location defined in on regular grid

	# find neighbour pixels on regular grid
	# x is horizontal, y is vertical
	img_inds = tf.to_int32(unknown[:,0])
	x = unknown[:,1]
	y = unknown[:,2]
	# rgb_inds = tf.to_int32(unknown[:,3])

	low_x = tf.to_int32(tf.floor(x))
	high_x = tf.to_int32(tf.ceil(x))
	low_y = tf.to_int32(tf.floor(y))
	high_y = tf.to_int32(tf.ceil(y))

	# measure the weights for neighbourhood average based on distance
	dist_low_x = tf.expand_dims(x - tf.to_float(low_x), axis=-1)
	dist_high_x = tf.expand_dims(tf.to_float(high_x) - x, axis=-1)
	dist_low_y = tf.expand_dims(y - tf.to_float(low_y), axis=-1)
	dist_high_y = tf.expand_dims(tf.to_float(high_y) - y, axis=-1)

	# compute horizontal avarage
	avg_low_y = dist_low_x*tf.gather_nd(data, indices=tf.stack([img_inds,low_y,low_x],axis=1)) + dist_high_x*tf.gather_nd(data, indices=tf.stack([img_inds,low_y,high_x],axis=1))
	avg_high_y = dist_low_x*tf.gather_nd(data, indices=tf.stack([img_inds,high_y,low_x],axis=1)) + dist_high_x*tf.gather_nd(data, indices=tf.stack([img_inds,high_y,high_x],axis=1))

	# compute vertical average
	avg = dist_low_y*avg_low_y + dist_high_y*avg_high_y

	return avg






