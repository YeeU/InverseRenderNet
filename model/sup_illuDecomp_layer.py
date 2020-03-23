import tensorflow as tf


# am is the albedo map, which has shape (batch, height, width, 3[rgb]) 
# nm is the sparse normal map, which has shape (batch, height, width, 3[x,y,z])
# L_SHcoeff contains the SH coefficients for environment illumination, using 2nd order SH. L_SHcoeff has shape (batch, 9, 3[rgb])
def illuDecomp(input, am, nm, gamma):

	""" 
	i = albedo * irradiance
	the multiplication is elementwise
	albedo is given
	irraidance = n.T * M * n, where n is (x,y,z,1)
	M is contructed from some precomputed constants and L_SHcoeff, where M contains information about illuminations, clamped cosine and SH basis
	"""

	# compute shading by dividing input by albedo
	shadings = tf.pow(input,gamma)/(am)
	# perform clamping on resulted shading to guarantee its numerical range
	shadings = tf.clip_by_value(shadings, 0., 1.) + tf.constant(1e-4)


	# compute shading by linear equation regarding nm and L_SHcoeffs
	c1 = tf.constant(0.429043,dtype=tf.float32)
	c2 = tf.constant(0.511664,dtype=tf.float32)
	c3 = tf.constant(0.743125,dtype=tf.float32)
	c4 = tf.constant(0.886227,dtype=tf.float32)
	c5 = tf.constant(0.247708,dtype=tf.float32)


	# find defined pixels
	mask = tf.not_equal(tf.reduce_sum(nm,axis=-1),0)
	num_iter = tf.shape(mask)[0]
	output = tf.TensorArray(dtype=tf.float32, size=num_iter)
	i = tf.constant(0)

	def condition(i, output):
		return i<num_iter

	def body(i, output):
		mask_ = mask[i]
		shadings_ = shadings[i]
		nm_ = nm[i]
		shadings_pixel = tf.boolean_mask(shadings_, mask_)
		nm_ = tf.boolean_mask(nm_, mask_)

		# E(n) = A*L_SHcoeffs
		total_npix = tf.shape(nm_)[0:1]
		ones = tf.ones(total_npix)
		A = tf.stack([c4*ones, 2*c2*nm_[:,1], 2*c2*nm_[:,2], 2*c2*nm_[:,0], 2*c1*nm_[:,0]*nm_[:,1], 2*c1*nm_[:,1]*nm_[:,2], c3*nm_[:,2]**2-c5, 2*c1*nm_[:,2]*nm_[:,0], c1*(nm_[:,0]**2-nm_[:,1]**2)], axis=-1)
		output = output.write(i, tf.matmul(pinv(A), shadings_pixel))
		i += tf.constant(1)

		return i, output

	_, output = tf.while_loop(condition, body, loop_vars=[i,output])
	L_SHcoeffs = output.stack()

	return tf.reshape(L_SHcoeffs, [-1,27])



def pinv(A, reltol=1e-6):
	# compute SVD of input A
	s, u, v = tf.svd(A)

	# invert s and clear entries lower than reltol*s_max
	atol = tf.reduce_max(s) * reltol
	s = tf.boolean_mask(s, s>atol)
	s_inv = tf.diag(1./s)

	# compute v * s_inv * u_t as psuedo inverse
	return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))



