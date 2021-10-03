import importlib
import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers


def SfMNet(inputs, height, width, name='', n_layers=12, n_pools=2, is_training=True, depth_base=64):
    conv_layers = np.int32(n_layers / 2) - 1
    deconv_layers = np.int32(n_layers / 2)
    # number of layers before perform pooling
    nlayers_befPool = np.int32(np.ceil((conv_layers - 1) / n_pools) - 1)

    max_depth = 512

    if depth_base * 2 ** n_pools < max_depth:
        tail = conv_layers - nlayers_befPool * n_pools

        tail_deconv = deconv_layers - nlayers_befPool * n_pools
    else:
        maxNum_pool = np.log2(max_depth / depth_base)
        tail = np.int32(conv_layers - nlayers_befPool * maxNum_pool)
        tail_deconv = np.int32(deconv_layers - nlayers_befPool * maxNum_pool)

    f_in_conv = [3] + [np.int32(depth_base * 2 ** (np.ceil(i / nlayers_befPool) - 1)) for i in
                       range(1, conv_layers - tail + 1)] + [np.int32(depth_base * 2 ** maxNum_pool) for i in
                                                            range(conv_layers - tail + 1, conv_layers + 1)]
    f_out_conv = [64] + [np.int32(depth_base * 2 ** (np.floor(i / nlayers_befPool))) for i in
                         range(1, conv_layers - tail + 1)] + [np.int32(depth_base * 2 ** maxNum_pool) for i in
                                                              range(conv_layers - tail + 1, conv_layers + 1)]

    f_in_deconv = f_out_conv[:0:-1] + [64]
    f_out_amDeconv = f_in_conv[:0:-1] + [3]
    f_out_MaskDeconv = f_in_conv[:0:-1] + [2]
    f_out_nmDeconv = f_in_conv[:0:-1] + [2]

    batch_norm_params = {'decay': 0.9, 'center': True, 'scale': True, 'epsilon': 1e-4,
                         'param_initializers': {'beta_initializer': tf.zeros_initializer(),
                                                'gamma_initializer': tf.ones_initializer(),
                                                'moving_variance_initializer': tf.ones_initializer(),
                                                'moving_average_initializer': tf.zeros_initializer()},
                         'param_regularizers': {'beta_regularizer': None,
                                                'gamma_regularizer': layers.l2_regularizer(scale=1e-5)},
                         'is_training': is_training, 'trainable': is_training}

    ### contractive conv_layer block
    conv_out = inputs
    conv_out_list = []
    for i, f_in, f_out in zip(range(1, conv_layers + 2), f_in_conv, f_out_conv):
        scope = name + 'conv' + str(i)

        if np.mod(i - 1, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool + 1 and i != 1:
            conv_out_list.append(conv_out)
            conv_out = layers.conv2d(conv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                     normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params,
                                     weights_initializer=tf.random_normal_initializer(mean=0,
                                                                                      stddev=np.sqrt(2 / 9 / f_in)),
                                     weights_regularizer=layers.l2_regularizer(scale=1e-5), biases_initializer=None,
                                     scope=scope, trainable=is_training)
            conv_out = tf.nn.max_pool(conv_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        else:

            conv_out = layers.conv2d(conv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                     normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params,
                                     weights_initializer=tf.random_normal_initializer(mean=0,
                                                                                      stddev=np.sqrt(2 / 9 / f_in)),
                                     weights_regularizer=layers.l2_regularizer(scale=1e-5), biases_initializer=None,
                                     scope=scope, trainable=is_training)

    ### expanding deconv_layer block succeeding conv_layer block
    am_deconv_out = conv_out
    for i, f_in, f_out in zip(range(1, deconv_layers + 1), f_in_deconv, f_out_amDeconv):
        scope = name + 'am/am_deconv' + str(i)

        # expand resolution every after nlayers_befPool deconv_layer
        if np.mod(i, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool:
            with tf.variable_scope(scope):
                W = tf.get_variable(regularizer=layers.l2_regularizer(scale=1e-5),
                                    initializer=get_bilinear_filter([3, 3, f_out, f_in], 2), shape=[3, 3, f_out, f_in],
                                    name='filter', trainable=is_training)
                # import ipdb; ipdb.set_trace()
                # attach previous convolutional output to upsampling/deconvolutional output
                tmp = conv_out_list[-np.int32(i / nlayers_befPool)]
                output_shape = tf.shape(tmp)
                am_deconv_out = tf.nn.conv2d_transpose(am_deconv_out, filter=W, output_shape=output_shape,
                                                       strides=[1, 2, 2, 1], padding='SAME')
                am_deconv_out = layers.batch_norm(scope=scope, activation_fn=tf.nn.relu, inputs=am_deconv_out,
                                                  decay=0.9, center=True, scale=True,
                                                  param_initializers={'beta_initializer': tf.zeros_initializer(),
                                                                      'gamma_initializer': tf.ones_initializer(),
                                                                      'moving_variance_initializer':
                                                                          tf.ones_initializer(),
                                                                      'moving_average_initializer':
                                                                          tf.zeros_initializer()},
                                                  param_regularizers={'beta_regularizer': None,
                                                                      'gamma_regularizer': layers.l2_regularizer(
                                                                          scale=1e-5)}, is_training=is_training,
                                                  trainable=is_training)

            tmp = layers.conv2d(tmp, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params,
                                weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / 9 / f_in)),
                                weights_regularizer=layers.l2_regularizer(scale=1e-5), biases_initializer=None,
                                scope=scope, trainable=is_training)
            am_deconv_out = tmp + am_deconv_out


        elif i == deconv_layers:
            am_deconv_out = layers.conv2d(am_deconv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1],
                                          padding='SAME', normalizer_fn=None, activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(
                                              2 / 9 / f_in)), weights_regularizer=layers.l2_regularizer(scale=1e-5),
                                          scope=scope, trainable=is_training)


        else:
            am_deconv_out = layers.conv2d(am_deconv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1],
                                          padding='SAME', normalizer_fn=layers.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(
                                              2 / 9 / f_in)), weights_regularizer=layers.l2_regularizer(scale=1e-5),
                                          biases_initializer=None, scope=scope, trainable=is_training)

    ### deconvolution net for nm estimates
    nm_deconv_out = conv_out
    for i, f_in, f_out in zip(range(1, deconv_layers + 1), f_in_deconv, f_out_nmDeconv):
        scope = name + 'nm/nm' + str(i)

        # expand resolution every after nlayers_befPool deconv_layer
        if np.mod(i, nlayers_befPool) == 0 and i <= n_pools * nlayers_befPool:
            with tf.variable_scope(scope):
                W = tf.get_variable(regularizer=layers.l2_regularizer(scale=1e-5),
                                    initializer=get_bilinear_filter([3, 3, f_out, f_in], 2), shape=[3, 3, f_out, f_in],
                                    name='filter', trainable=is_training)

                # attach previous convolutional output to upsampling/deconvolutional output
                tmp = conv_out_list[-np.int32(i / nlayers_befPool)]
                output_shape = tf.shape(tmp)
                nm_deconv_out = tf.nn.conv2d_transpose(nm_deconv_out, filter=W, output_shape=output_shape,
                                                       strides=[1, 2, 2, 1], padding='SAME')
                nm_deconv_out = layers.batch_norm(scope=scope, activation_fn=tf.nn.relu, inputs=nm_deconv_out,
                                                  decay=0.9, center=True, scale=True, epsilon=1e-4,
                                                  param_initializers={'beta_initializer': tf.zeros_initializer(),
                                                                      'gamma_initializer': tf.ones_initializer(),
                                                                      'moving_variance_initializer': tf.ones_initializer(),
                                                                      'moving_average_initializer': tf.zeros_initializer()},
                                                  param_regularizers={'beta_regularizer': None,
                                                                      'gamma_regularizer': layers.l2_regularizer(
                                                                          scale=1e-5)}, is_training=is_training,
                                                  trainable=is_training)

            tmp = layers.conv2d(tmp, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                normalizer_fn=layers.batch_norm, normalizer_params=batch_norm_params,
                                weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(2 / 9 / f_in)),
                                weights_regularizer=layers.l2_regularizer(scale=1e-5), biases_initializer=None,
                                scope=scope, trainable=is_training)
            nm_deconv_out = tmp + nm_deconv_out


        elif i == deconv_layers:
            nm_deconv_out = layers.conv2d(nm_deconv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1],
                                          padding='SAME', normalizer_fn=None, activation_fn=None,
                                          weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(
                                              2 / 9 / f_in)), weights_regularizer=layers.l2_regularizer(scale=1e-5),
                                          biases_initializer=None, scope=scope, trainable=is_training)


        else:
            nm_deconv_out = layers.conv2d(nm_deconv_out, num_outputs=f_out, kernel_size=[3, 3], stride=[1, 1],
                                          padding='SAME', normalizer_fn=layers.batch_norm,
                                          normalizer_params=batch_norm_params,
                                          weights_initializer=tf.random_normal_initializer(mean=0, stddev=np.sqrt(
                                              2 / 9 / f_in)), weights_regularizer=layers.l2_regularizer(scale=1e-5),
                                          biases_initializer=None, scope=scope, trainable=is_training)

    return am_deconv_out, nm_deconv_out


def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    bilinear = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
    weights = np.tile(bilinear[:, :, None, None], (1, 1, filter_shape[2], filter_shape[3]))

    return tf.constant_initializer(weights)
