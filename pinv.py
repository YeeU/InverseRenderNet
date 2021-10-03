import tensorflow as tf


def pinv(A, reltol=1e-6):
    """compute pseudo inverse for input matrix"""
    # compute SVD of input A
    s, u, v = tf.svd(A)

    # invert s and clear entries lower than reltol*s_max
    atol = tf.reduce_max(s) * reltol
    s = tf.boolean_mask(s, s > atol)
    s_inv = tf.diag(1. / s)

    # compute v * s_inv * u_t as psuedo inverse
    return tf.matmul(v, tf.matmul(s_inv, tf.transpose(u)))
