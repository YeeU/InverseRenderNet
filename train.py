# also predict shadow mask and error mask

# no rotation


#### compute albedo reproj loss only on reprojection available area; compute reconstruction and its loss only based
# on defined area


import tensorflow as tf
import importlib
import os
import pickle as pk
import sys
import numpy as np
import time
import argparse
from PIL import Image
import glob
from model import SfMNet, lambSH_layer, pred_illuDecomp_layer, loss_layer, dataloader

parser = argparse.ArgumentParser(description='InverseRenderNet')
parser.add_argument('--n_batch', '-n', help='number of minibatch', type=int)
parser.add_argument('--data_path', '-p', help='Path to training data')
parser.add_argument('--train_mode', '-m', help='specify the phase for training (pre-train/self-train)',
                    choices={'pre-train', 'self-train'})

args = parser.parse_args()


def main():
    inputs_shape = (5, 200, 200, 3)

    next_element, trainData_init_op, num_train_batches = dataloader.megaDepth_dataPipeline(args.n_batch, args.data_path)

    inputs_var = tf.reshape(next_element[0], (-1, inputs_shape[1], inputs_shape[2], inputs_shape[3]))
    dms_var = tf.reshape(next_element[1], (-1, inputs_shape[1], inputs_shape[2]))
    nms_var = tf.reshape(next_element[2], (-1, inputs_shape[1], inputs_shape[2], 3))
    cams_var = tf.reshape(next_element[3], (-1, 16))
    scaleXs_var = tf.reshape(next_element[4], (-1,))
    scaleYs_var = tf.reshape(next_element[5], (-1,))
    masks_var = tf.reshape(next_element[6], (-1, inputs_shape[1], inputs_shape[2]))

    # var helping cross projection
    pair_label_var = tf.constant(np.repeat(np.arange(args.n_batch), inputs_shape[0])[:, None], dtype=tf.float32)
    # weights for smooth loss and am_consistency loss
    am_smt_w_var = tf.placeholder(tf.float32, ())
    reproj_w_var = tf.placeholder(tf.float32, ())

    # mask out sky in inputs and nms
    masks_var_4d = tf.expand_dims(masks_var, axis=-1)
    inputs_var *= masks_var_4d
    nms_var *= masks_var_4d

    # inverserendernet
    if args.train_mode == 'pre-train':
        am_deconvOut, nm_deconvOut = SfMNet.SfMNet(inputs=inputs_var, is_training=True, height=inputs_shape[1],
                                                   width=inputs_shape[2], name='pre_train_IRN/', n_layers=30, n_pools=4,
                                                   depth_base=32)

        am_sup = tf.zeros_like(am_deconvOut)
        preTrain_flag = True


    elif args.train_mode == 'self-train':
        am_deconvOut, nm_deconvOut = SfMNet.SfMNet(inputs=inputs_var, is_training=True, height=inputs_shape[1],
                                                   width=inputs_shape[2], name='IRN/', n_layers=30, n_pools=4,
                                                   depth_base=32)

        am_sup, _ = SfMNet.SfMNet(inputs=inputs_var, is_training=False, height=inputs_shape[1], width=inputs_shape[2],
                                  name='pre_train_IRN/', n_layers=30, n_pools=4, depth_base=32)
        am_sup = tf.nn.sigmoid(am_sup) * masks_var_4d + tf.constant(1e-4)

        preTrain_flag = False

    # separate albedo, error mask and shadow mask from deconvolutional output
    albedoMaps = am_deconvOut[:, :, :, :3]

    # formulate loss
    light_SHCs, albedoMaps, nm_preds, loss, render_err, reproj_err, cross_render_err, reg_loss, illu_prior_loss, \
	albedo_smt_error, nm_smt_loss, nm_loss, am_loss = loss_layer.loss_formulate(
        albedoMaps, nm_deconvOut, am_sup, nms_var, inputs_var, dms_var, cams_var, scaleXs_var, scaleYs_var,
        masks_var_4d, pair_label_var, True, am_smt_w_var, reproj_w_var, reg_loss_flag=True)

    # defined traning loop
    epochs = 30
    num_batches = num_train_batches
    num_subbatch = args.n_batch
    num_iters = np.int32(np.ceil(num_batches / num_subbatch))

    # training op
    global_step = tf.Variable(1, name='global_step', trainable=False)

    train_step = tf.contrib.layers.optimize_loss(loss,
                                                 optimizer=tf.train.AdamOptimizer(learning_rate=.05, epsilon=1e-1),
                                                 learning_rate=None, global_step=global_step)

    # define saver for saving and restoring
    irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope='IRN') if args.train_mode == 'self-train' else tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='pre_train_IRN')
    saver = tf.train.Saver(irn_vars)

    # define session
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

    # train from scratch or keep training trained model
    tf.local_variables_initializer().run()
    tf.global_variables_initializer().run()

    assignOps = []
    if args.train_mode == 'self-train':
        # load am_sup net
        preTrain_irn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pre_train_IRN')
        saver_loadOldVar = tf.train.Saver(preTrain_irn_vars)
        saver_loadOldVar.restore(sess, 'pre_train_model/model.ckpt')

        # import ipdb; ipdb.set_trace()
        # duplicate pre_train model
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            vars = tf.contrib.framework.list_variables('pre_train_model')
            for var_name, _ in vars:
                var = tf.contrib.framework.load_variable('pre_train_model', var_name)
                new_var_name = var_name.replace('pre_train_IRN', 'IRN')

                new_var = tf.get_variable(name=new_var_name)
                assignOps += [new_var.assign(var)]

        sess.run(assignOps)

    # start training
    trainData_init_op.run()
    dst_dir = 'irn_model' if args.train_mode == 'self-train' else 'pre_train_model'
    for i in range(1, epochs + 1):

        loss_avg = 0
        f = open('cost.txt', 'a')

        # graduately update weights if pre-training
        reproj_weight = .2 + np.clip(.8 * (i - 16) / 14, 0., .8) if args.train_mode == 'pre-train' else 1.
        am_smt_weight = .2 + np.clip(.8 * (i - 1) / 14, 0., .8) if args.train_mode == 'pre-train' else 1.

        for j in range(1, num_iters + 1):
            start_time = time.time()

            # train
            [loss_val, reg_loss_val, render_err_val, reproj_err_val, cross_render_err_val, illu_prior_val,
             albedo_smt_error_val, nm_smt_loss_val, nm_loss_val, am_loss_val] = sess.run(
                [train_step, reg_loss, render_err, reproj_err, cross_render_err, illu_prior_loss, albedo_smt_error,
                 nm_smt_loss, nm_loss, am_loss], feed_dict={am_smt_w_var: am_smt_weight, reproj_w_var: reproj_weight})
            loss_avg += loss_val

            # log
            if j % 1 == 0:
                print('iter %d/%d loop %d/%d took %.3fs' % (i, epochs, j, num_iters, time.time() - start_time))
                print('\tloss_avg = %f, loss = %f' % (loss_avg / j, loss_val))
                print(
                    '\t\treg_loss = %f, render_err = %f, reproj_err = %f, cross_render_err = %f, illu_prior = %f, albedo_smt_error = %f, nm_smt_loss = %f, nm_loss = %f, am_loss = %f' % (
                    reg_loss_val, render_err_val, reproj_err_val, cross_render_err_val, illu_prior_val,
                    albedo_smt_error_val, nm_smt_loss_val, nm_loss_val, am_loss_val))

                f.write(
                    'iter %d/%d loop %d/%d took %.3fs\n\tloss_avg = %f, loss = %f\n\t\treg_loss = %f, render_err = %f, reproj_err = %f, cross_render_err = %f, illu_prior = %f, albedo_smt_error = %f, nm_smt_loss = %f, nm_loss = %f, am_loss = %f\n' % (
                    i, epochs, j, num_iters, time.time() - start_time, loss_avg / j, loss_val, reg_loss_val,
                    render_err_val, reproj_err_val, cross_render_err_val, illu_prior_val, albedo_smt_error_val,
                    nm_smt_loss_val, nm_loss_val, am_loss_val))

        f.close()

        # save model every 10 iterations
        saver.save(sess, os.path.join(dst_dir, 'model.ckpt'))


if __name__ == '__main__':
    main()
