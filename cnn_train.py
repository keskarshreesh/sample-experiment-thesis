"""
Train models using multiple GPU's with synchronous updates.
Adapted from inception_train.py

This is modular, i.e. it is not tied to any particular
model or dataset.

@Author: Ankush Gupta, Tomas Jakab
@Date: 25 Aug 2016
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
from datetime import datetime
import os.path as osp
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import json
import os
import random
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python import debug as tf_debug
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

from colorize import colorize
from log_tensorboard import log_tensorboard
from model import model
from config import opts


def get_train_summaries():
  summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
  return summaries


def get_bnorm_ops():
    """
    Return any batch-normalization / other "moving-average" ops.
    ref: https://github.com/tensorflow/tensorflow/issues/1122#issuecomment-236068575
    """
    updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # print updates
    return tf.group(*updates)


def tower_loss(inputs, flag, training_pl):
  """
  Args:
    images: Images. 4D tensor of size [batch_size,H,W,C].
    labels: 1-D integer Tensor of [batch_size,EXTRA_DIMS (optional)].
    model: object which defines the model. Needs to have a `build` function.
    scope: unique prefix string identifying the tower, e.g. 'tower_0'.

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  # Build Graph. Note,we force the variables to lie on the CPU,
  # required for multi-gpu training (automatically placed):
  loss_dict, ret_dict = model(inputs, flag, training_pl, costs_collection='costs')
    
  return loss_dict, ret_dict


def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def train_single(graph, inputs, optim1, flag, training_pl, global_step, clip_value=None):
  """
  Train on dataset for a number of steps.
  Args:
    opts: dict, dictionary with the following options:
      gpu_ids: list of integer indices of the GPUs to use
      batch_size: integer: total batch size
                  (each GPU processes batch_size/num_gpu instances)
    graph: tf.Graph instance
    model_factory: function which creates TFModels.
                   Multiple such models are created
                   for each GPU.
                   create_optimizer(lr): returns an optimizer
  """

  with graph.as_default():
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Calculate the gradients for each model tower.

    loss_dict, ret_dict = tower_loss(inputs, flag, training_pl)

    # summaries and batch-norm updates:
    train_summaries = get_train_summaries()
    # test_summaries = get_test_summaries()
    bnorm_updates = get_bnorm_ops()

    train_vars = tf.trainable_variables()
    print ("all trainable variables ")
    for i in train_vars:
      print (i)

    train_vars1 = []
    count = 46
    while count < 56:
      # train_vars.pop(0)
      train_vars1.append(train_vars[count])
      count += 1

    # count = 98
    # while count < 156:
    #   # train_vars.pop(0)
    #   train_vars1.append(train_vars[count])
    #   count += 1

    # print ("actual trainable variables ")
    # for i in train_vars1:
    #   print (i)

    # param_cam_angles = get_network_params(scope='pose_encoder/cam_branch/cam_angles')
    # print ("actual trainable variables ")
    # for i in param_cam_angles:
    #   print (i)

    grads_and_vars1 = optim1.compute_gradients(loss_dict['recons_loss']+loss_dict['loss_2d'], var_list=train_vars)

    apply_grad_op1 = optim1.apply_gradients(grads_and_vars1, global_step=global_step)


    # Group all updates to into a single train op:
    # train_op = tf.group(apply_grad_op, apply_grad_op_3d, bnorm_updates)
    train_op1 = tf.group(apply_grad_op1, bnorm_updates)

    # Add a summaries for the input processing and global_step.
    train_summaries.extend(input_summaries)
    # test_summaries.extend(input_summaries)
    train_summary_op = tf.summary.merge(train_summaries)
    # test_summary_op = tf.summary.merge(test_summaries)
    test_summary_op = train_summary_op

    return loss_dict, ret_dict, train_op1, train_summary_op, test_summary_op


def train_loop(graph, loss_dict, ret_dict, mads_dataset, human_dataset, inputs, training_pl, handle_pl, 
               train_op1, flag, train_summary_op, test_summary_op,
               num_steps, global_step, test_dataset=None):

  tf.logging.set_verbosity(tf.logging.INFO)
  with graph.as_default(), tf.device('/cpu:0'):
    # define iterators
    mads_iterator = mads_dataset.make_initializable_iterator()
    human_iterator = human_dataset.make_initializable_iterator()
    # if test_dataset:
    #   test_iterator = test_dataset.make_initializable_iterator()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    session_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    # h36 = 0
    mads_unsup = 1
    restore = True
    test_only = False
    fwd_only = False
    begin_time = time.time()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    with tf.Session(config=session_config) as sess:

      ######## initialize
      global_init = tf.global_variables_initializer()
      local_init = tf.local_variables_initializer()
      print ("global_init")
      print (global_init)
      print ("local_init")
      print (local_init)
      sess.run([global_init,local_init])


      ######## restore
      if restore == True:
        try:
          vars_to_restore1 = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)
          vars_to_restore2 = tf.global_variables()
          print ("Total vars_to_restore1 ", vars_to_restore1)
          print ("Total vars_to_restore2 ", vars_to_restore2)

          # pose
          vars_to_restore = []
          for v in vars_to_restore2:
            # if 'app_encoder' not in v.name and 'unified_decoder' not in v.name and 'bg_append' not in v.name:
              vars_to_restore.append(v)
          print ("vars_to_restore2 ", vars_to_restore)
          restorer = tf.train.Saver(var_list=vars_to_restore)
          print(colorize('vars-to-be-restored:','green',bold=True))
          print(colorize(', '.join([v.name for v in vars_to_restore]),'green'))
          print(colorize('Trying to RESTORE MODEL', 'blue', bold=True))
          restorer.restore(sess, tf.train.latest_checkpoint('./log_dir/checkpoints'))

        except Exception as e:
          print ("Error loading model.....Training from scratch! ", e)

      # try restoring resnet_v1_50 weights
      # init_fn = slim.assign_from_checkpoint_fn(os.path.join('/data/vcl/sid/saved_models/resnet_checkpoints_dir', 'resnet_v1_50.ckpt'), slim.get_model_variables('resnet_v1_50'))
      # init_fn(sess)
      
      ######## set up iterators
      mads_unsup_handle = sess.run(mads_iterator.string_handle())
      mads_sup_handle = sess.run(human_iterator.string_handle())
      sess.run(mads_iterator.initializer)
      sess.run(human_iterator.initializer)
      # if test_dataset:
      #   test_handle = sess.run(test_iterator.string_handle())
      # sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root="./tmp")

      ######## create a summary writer:
      summary_writer = tf.summary.FileWriter(opts['log_dir'], graph=sess.graph)

      ######## get the value of the global-step:
      start_step = sess.run(global_step)
      
      if test_only:
        print(colorize('Only testing............................................','green',bold=True))
      else:
        print ("training loop starting")

      ######## run the training loop:
      for step in range(int(start_step), int(num_steps)):
        try:
          start_time = time.time()

          batch_size = opts['batch_size']

          # try:
          
          if test_only:
              feed_dict = {handle_pl: train_handle, training_pl: True}
              recons_loss_val, return_dict, summary_str = sess.run([loss_dict['recons_loss'], ret_dict, train_summary_op], feed_dict=feed_dict)

          else:
              if mads_unsup:
                feed_dict = {handle_pl: mads_unsup_handle, training_pl: True}
                mads_unsup = 0
              else:
                feed_dict = {handle_pl: mads_sup_handle, training_pl: True}
                mads_unsup = 1
              recons_loss_val, return_dict, _, summary_str = sess.run([loss_dict['recons_loss'], ret_dict, train_op1, train_summary_op], feed_dict=feed_dict)

          # except Exception as e:
          #   print ("e in train sk_3d ", e)


          duration = time.time() - start_time
          examples_per_sec = opts['batch_size'] / float(duration)
          assert not np.isnan(recons_loss_val), 'Model diverged with loss = NaN'
          format_str = '%s: step %d, loss_recons = %.4f (%.1f examples/sec) %.3f sec/batch'
          tf.logging.info(format_str % (datetime.now(), step, recons_loss_val, examples_per_sec, duration))

          if step % (opts['n_summary']+1) == 0:
            try:
              print (return_dict['projs_skeleton_2d'][0])
              print (return_dict['unscaled_projs_skeleton_2d'][0])
              print (return_dict['focal_length'][0])
              print (return_dict['out_cam_params'][0])
              print (return_dict['out_cam_angles'][0])
              log_tensorboard(return_dict, summary_writer, step, 1-mads_unsup)
              summary_writer.add_summary(summary_str, step)
              summary_writer.flush() # write to disk now
            except Exception as e:
              print ("e in plotting ", e)

          # periodically write the summary (after every N_SUMMARY steps):
          # periodically checkpoint:
          if step % opts['n_checkpoint'] == 0:
            try:
              checkpoint_path = osp.join(opts['log_dir'], 'model')
              saver.save(sess, checkpoint_path, global_step=step)
            except Exception as e:
              print ("e in saving ", e)
        except:
          sess.run(mads_iterator.initializer)
          sess.run(human_iterator.initializer)
      total_time = time.time()-begin_time
      samples_per_sec = opts['batch_size'] * num_steps / float(total_time)
      print('Avg. samples per second %.3f'%samples_per_sec)
