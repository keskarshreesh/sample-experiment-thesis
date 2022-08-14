# ==========================================================
# Author: Siddharth Seth
# ==========================================================
from __future__ import print_function
from __future__ import absolute_import
import os,sys
import json
import time
from tensorflow.contrib.framework.python.ops import variables
import tensorflow as tf
import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import imutils

import cnn_train as tru
from colorize import colorize
from data_loader import _read_py_function1, _read_py_function2, read_dummy_list
from config import opts


def main(args):

  # get the data and logging (checkpointing) directories:
  log_dir = 'log_dir'

  NUM_STEPS = 3000000


  graph = tf.Graph()
  with graph.as_default():
    global_step = variables.model_variable('global_step',shape=[],
                                            initializer=tf.constant_initializer(args.reset_global_step),
                                            trainable=False)


    batch_size = opts['batch_size']
    lr = tf.train.exponential_decay(0.001, global_step, 100000, 0.95, staircase=True)
    # tf.summary.scalar('lr', lr) # add a summary
    # common model / optimizer parameters:
    # optim1 = tf.train.RMSPropOptimizer(lr, name='RMS')
    # optim1 = tf.train.AdamOptimizer(lr, name='Adam')
    optim1 = tf.train.AdagradOptimizer(lr, use_locking=False,name='AdaGrad')
    # optim2 = tf.train.AdagradOptimizer(lr, use_locking=False,name='AdaGrad')

    print(colorize('log_dir: ' + log_dir,'green',bold=True))
    print(colorize('BATCH-SIZE: %d'%batch_size,'red',bold=True))

    dummy_mads_data, dummy_human_data = read_dummy_list()

    mads_dataset = tf.data.Dataset.from_tensor_slices((dummy_mads_data))
    human_dataset = tf.data.Dataset.from_tensor_slices((dummy_human_data))
    
    # train_dataset = train_dataset.shuffle(len(current_train_data))
    # test_dataset = test_dataset.shuffle(len(current_test_data))

    num_parallel_calls = 16

    mads_dataset = mads_dataset.map(lambda z: tf.py_func(_read_py_function1, [z], [tf.float32, tf.float32, tf.float32, tf.float32, 
                                                                                    tf.float32, tf.float32]), num_parallel_calls=num_parallel_calls)
    human_dataset = human_dataset.map(lambda z: tf.py_func(_read_py_function2, [z], [tf.float32, tf.float32, tf.float32, tf.float32, 
                                                                                    tf.float32, tf.float32]), num_parallel_calls=num_parallel_calls)
 
    print ("outside pyfunc")
    mads_dataset = mads_dataset.batch(batch_size)
    human_dataset = human_dataset.batch(batch_size)
    mads_dataset = mads_dataset.prefetch(2)
    human_dataset = human_dataset.prefetch(2)

    mads_dset = mads_dataset
    human_dset = human_dataset

    print ("reached training")

    # set up inputs
    training_pl = tf.placeholder(tf.bool)
    flag = tf.placeholder(tf.int32, shape=[], name='flag')
    handle_pl = tf.placeholder(tf.string, shape=[])


    base_iterator = tf.data.Iterator.from_string_handle(handle_pl, mads_dset.output_types, mads_dset.output_shapes)

    src_im, future_im, label_im, back_im, pose_2d, loss_flag = base_iterator.get_next()
    inputs = {'source_im': src_im, 'future_im': future_im, 'label_im': label_im, 'back_im': back_im, 'pose_2d': pose_2d, 'loss_flag': loss_flag} #, 'label': label}

    split_gpus = False
    print ("inputs = ", inputs)
    print ("type(inputs) ", type(inputs))
    # create the network distributed over multi-GPUs:
    loss_dict, ret_dict, train_op1, train_summary_op, test_summary_op = tru.train_single(graph, inputs, optim1, flag, training_pl, global_step)

    # run the training loop:
    if args.restore_optim:
      restore_vars = 'all'
    else:
      restore_vars = 'model'

    tru.train_loop(graph, loss_dict, ret_dict, mads_dset, human_dset, inputs, training_pl, handle_pl, 
                    train_op1, flag, train_summary_op, test_summary_op, NUM_STEPS, global_step)

if  __name__=='__main__':
  import argparse
  parser = argparse.ArgumentParser(description='Train Unsupervised Sequence Model')
  parser.add_argument('--configs', nargs='+', default=[], help='Paths to the config files.')
  parser.add_argument('--ngpus',type=int,default=1,required=False,help='Number of GPUs to use for training.')
  parser.add_argument('--lr-multiple',type=float,default=1,help='multiplier on the learning rate.')
  parser.add_argument('--checkpoint',type=str,default=None,
                      help='checkpoint file-name of the *FULL* model to restore.')
  parser.add_argument('--restore-optim',action='store_true',help='Restore the optimizer variables.')
  parser.add_argument('--reset-global-step',type=int,default=-1,help='Force the value of global step.')
  parser.add_argument('--ignore-missing-vars',action='store_true',help='Skip re-storing vars not in the checkpoint file.')
  args = parser.parse_args()
  main(args)
