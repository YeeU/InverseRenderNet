import tensorflow as tf


# am is the albedo map, which has shape (batch, height, width, 3[rgb]) 
# nm is the sparse normal map, which has shape (batch, height, width, 3[x,y,z])
# L_SHcoeff contains the SH coefficients for environment illumination, using 2nd order SH. L_SHcoeff has shape (batch, 9, 3[rgb])
def lambSH_layer(am, nm, L_SHcoeffs, gamma):

	""" 
	i = albedo * irradiance
	the multiplication is elementwise
	albedo is given
	irraidance = n.T * M * n, where n is (x,y,z,1)
	M is contructed from some precomputed constants and L_SHcoeff, where M contains information about illuminations, clamped cosine and SH basis
	"""

	# M is only related with lighting
	c1 = tf.constant(0.429043,dtype=tf.float32)
	c2 = tf.constant(0.511664,dtype=tf.float32)
	c3 = tf.constant(0.743125,dtype=tf.float32)
	c4 = tf.constant(0.886227,dtype=tf.float32)
	c5 = tf.constant(0.247708,dtype=tf.float32)

	# each row have shape (batch, 4, 3)
	M_row1 = tf.stack([c1*L_SHcoeffs[:,8,:], c1*L_SHcoeffs[:,4,:], c1*L_SHcoeffs[:,7,:], c2*L_SHcoeffs[:,3,:]],axis=1)
	M_row2 = tf.stack([c1*L_SHcoeffs[:,4,:], -c1*L_SHcoeffs[:,8,:], c1*L_SHcoeffs[:,5,:], c2*L_SHcoeffs[:,1,:]],axis=1)
	M_row3 = tf.stack([c1*L_SHcoeffs[:,7,:], c1*L_SHcoeffs[:,5,:], c3*L_SHcoeffs[:,6,:], c2*L_SHcoeffs[:,2,:]],axis=1)
	M_row4 = tf.stack([c2*L_SHcoeffs[:,3,:], c2*L_SHcoeffs[:,1,:], c2*L_SHcoeffs[:,2,:], c4*L_SHcoeffs[:,0,:]-c5*L_SHcoeffs[:,6,:]],axis=1)

	# M is a 5d tensot with shape (batch,4,4,3[rgb]), the axis 1 and 2 are transposely equivalent
	M = tf.stack([M_row1,M_row2,M_row3,M_row4], axis=1)

	# find batch-spatial three dimensional mask of defined normals over nm
	# mask = tf.logical_not(tf.is_nan(nm[:,:,:,0]))
	mask = tf.not_equal(tf.reduce_sum(nm,axis=-1),0)


	# extend Cartesian to homogeneous coords and extend its last for rgb individual multiplication dimension, nm_homo have shape (total_npix, 4)
	total_npix = tf.shape(nm)[:3]
	ones = tf.ones(total_npix)
	nm_homo = tf.concat([nm,tf.expand_dims(ones,axis=-1)], axis=-1)

	# contruct batch-wise flatten M corresponding with nm_homo, such that multiplication between them is batch-wise
	M = tf.expand_dims(tf.expand_dims(M,axis=1),axis=1)


	# expand M for broadcasting, such that M has shape (npix,4,4,3)
	# expand nm_homo, such that nm_homo has shape (npix,4,1,1)
	nm_homo = tf.expand_dims(tf.expand_dims(nm_homo,axis=-1),axis=-1)
	# tmp have shape (npix, 4, 3[rgb])
	tmp = tf.reduce_sum(nm_homo*M,axis=-3)
	# E has shape (npix, 3[rbg])
	E = tf.reduce_sum(tmp*nm_homo[:,:,:,:,0,:],axis=-2)


	# compute intensity by product between irradiance and albedo
	i = E*am

	# gamma correction
	i = tf.clip_by_value(i, 0., 1.) + tf.constant(1e-4)
	i = tf.pow(i,1./gamma)

	return i, mask













