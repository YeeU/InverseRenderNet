import tensorflow as tf

from pinv import pinv


# am is the albedo map, which has shape (batch, height, width, 3[rgb])
# nm is the sparse normal map, which has shape (batch, height, width, 3[x,y,z])
# L_SHcoeff contains the SH coefficients for environment illumination, using 2nd order SH. L_SHcoeff has shape (
# batch, 9, 3[rgb])
def illuDecomp(input, am, nm, gamma, masks):
    """
    i = albedo * irradiance
    the multiplication is elementwise
    albedo is given
    irraidance = n.T * M * n, where n is (x,y,z,1)
    M is contructed from some precomputed constants and L_SHcoeff, where M contains information about illuminations,
    clamped cosine and SH basis
    """

    # compute shading by dividing input by albedo
    shadings = tf.pow(input, gamma) / am
    # perform clamping on resulted shading to guarantee its numerical range
    shadings = (tf.clip_by_value(shadings, 0., 1.) + tf.constant(1e-4)) * masks

    # compute shading by linear equation regarding nm and L_SHcoeffs
    # E(n) = c1*L22*(x**2-y**2) + (c3*z**2 - c5)*L20 + c4*L00 + 2*c1*L2-2*x*y + 2*c1*L21*x*z + 2*c1*L2-1*y*z +
    # 2*c2*L11*x + 2*c2*L1-1*y + 2*c2*L10*z
    # E(n) = c4*L00 + 2*c2*y*L1-1 + 2*c2*z*L10 + 2*c2*x*L11 + 2*c1*x*y*L2-2 + 2*c1*y*z*L2-1 + (c3*z**2 - c5)*L20 +
    # 2*c1*x*z*L21 + c1*(x**2-y**2)*L22
    c1 = tf.constant(0.429043, dtype=tf.float32)
    c2 = tf.constant(0.511664, dtype=tf.float32)
    c3 = tf.constant(0.743125, dtype=tf.float32)
    c4 = tf.constant(0.886227, dtype=tf.float32)
    c5 = tf.constant(0.247708, dtype=tf.float32)

    # find defined pixels
    num_iter = tf.shape(nm)[0]
    output = tf.TensorArray(dtype=tf.float32, size=num_iter)
    i = tf.constant(0)

    def condition(i, output):
        return i < num_iter

    def body(i, output):
        shadings_ = shadings[i]
        nm_ = nm[i]
        shadings_pixel = tf.reshape(shadings_, (-1, 3))
        nm_ = tf.reshape(nm_, (-1, 3))

        # E(n) = A*L_SHcoeffs
        total_npix = tf.shape(nm_)[0:1]
        ones = tf.ones(total_npix)
        A = tf.stack(
            [c4 * ones, 2 * c2 * nm_[:, 1], 2 * c2 * nm_[:, 2], 2 * c2 * nm_[:, 0], 2 * c1 * nm_[:, 0] * nm_[:, 1],
             2 * c1 * nm_[:, 1] * nm_[:, 2], c3 * nm_[:, 2] ** 2 - c5, 2 * c1 * nm_[:, 2] * nm_[:, 0],
             c1 * (nm_[:, 0] ** 2 - nm_[:, 1] ** 2)], axis=-1)
        output = output.write(i, tf.matmul(pinv(A), shadings_pixel))
        i += tf.constant(1)

        return i, output

    _, output = tf.while_loop(condition, body, loop_vars=[i, output])
    L_SHcoeffs = output.stack()

    return tf.reshape(L_SHcoeffs, [-1, 27])
