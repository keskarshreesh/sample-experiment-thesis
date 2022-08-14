"""
Network definitions
"""

import tensorflow as tf
import tensorflow.contrib.layers as tf_layer
import math
from collections import OrderedDict
from transformation import unit_norm_tf, make_skeleton
from config import opts
from misc.layers import *
import numpy as np

def app_encoder(img, reuse=False):
    with tf.variable_scope('app_encoder', reuse=reuse):
        batch_norm = True
        
        img = tf.reshape(img, (-1, img.shape[1], img.shape[2], img.shape[3]))
        conv1 = tf.layers.conv2d(img, 32, 7, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)

        conv2 = tf.layers.conv2d(relu1, 32, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        conv3 = tf.layers.conv2d(relu2, 64, 3, strides=2, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn3 = tf.layers.batch_normalization(conv3, training=batch_norm, fused=True)
        relu3 = tf.nn.relu(bn3)
        
        conv4 = tf.layers.conv2d(relu3, 64, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn4 = tf.layers.batch_normalization(conv4, training=batch_norm, fused=True)
        relu4 = tf.nn.relu(bn4)                             # 64x64x64

        conv9 = tf.layers.conv2d(relu4, 128, 3, strides=2, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn9 = tf.layers.batch_normalization(conv9, training=batch_norm, fused=True)
        relu9 = tf.nn.relu(bn9)
            
        conv10 = tf.layers.conv2d(relu9, 128, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn10 = tf.layers.batch_normalization(conv10, training=batch_norm, fused=True)
        relu10 = tf.nn.relu(bn10)                       # 56x56x128

        conv13 = tf.layers.conv2d(relu10, 256, 3, strides=2, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn13 = tf.layers.batch_normalization(conv13, training=batch_norm, fused=True)
        relu13 = tf.nn.relu(bn13)
        
        conv14 = tf.layers.conv2d(relu13, 256, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn14 = tf.layers.batch_normalization(conv14, training=batch_norm, fused=True)
        relu14 = tf.nn.relu(bn14)                                       # 28x28x256

        conv15 = tf.layers.conv2d(relu14, 512, 3, strides=2, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn15 = tf.layers.batch_normalization(conv15, training=batch_norm, fused=True)
        relu15 = tf.nn.relu(bn15)

        conv16 = tf.layers.conv2d(relu15, 512, 3, strides=1, padding='SAME', \
                                                  kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn16 = tf.layers.batch_normalization(conv16, training=batch_norm, fused=True)
        relu16 = tf.nn.relu(bn16)                                       # 14x14x512

        if opts['channel_wise_fc']:
        # # https://stackoverflow.com/questions/47556001/how-to-efficiently-apply-a-channel-wise-fully-connected-layer-in-tensorflow
            def channel_wise_fc_layer(input, name): # bottom: (7x7x512)
                _, width, height, n_feat_map = input.get_shape().as_list()
                input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
                input_transpose = tf.transpose( input_reshape, [2,0,1] )

                with tf.variable_scope(name):
                    W = tf.get_variable(
                            "W",
                            shape=[n_feat_map,width*height, width*height], # (512,49,49)
                            initializer=tf.random_normal_initializer(0., 0.005))
                    output = tf.matmul(input_transpose, W)

                output_transpose = tf.transpose(output, [1,2,0])
                output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

                return output_reshape

            conv17 = tf.layers.conv2d(relu16, 512, 3, strides=2, padding='SAME', \
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn17 = tf.layers.batch_normalization(conv17, training=batch_norm, fused=True)
            relu17 = tf.nn.relu(bn17)

            ch_fc = channel_wise_fc_layer(relu17, 'channel_fc')
            relu18 = tf.nn.relu(ch_fc)

            upsample1 = tf.image.resize_images(relu18, [14, 14])
            conv18 = tf.layers.conv2d(upsample1, 256, 3, strides=1, padding='SAME', \
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bn18 = tf.layers.batch_normalization(conv18, training=batch_norm, fused=True)
            app_embed = tf.nn.relu(bn18)
        else:
            app_embed = relu16

        return app_embed


def pose_encoder(img_features, reuse=False):
     with tf.variable_scope('pose_encoder', reuse=reuse):
        batch_norm = True
        ret_dict = {}
        print ("img_features ", img_features)
        flatten = tf.reshape(img_features, (-1, img_features.shape[1]*img_features.shape[2]*img_features.shape[3]))     # Bx14x14x512

        alpha = 0.1
        with tf.variable_scope('cam_branch', reuse=reuse):
            fc_1 = tf.layers.dense(flatten, 1024, activation=None, name='fc_1')
            bn1 = tf.layers.batch_normalization(fc_1, training=batch_norm, fused=True)
            lr1 = tf.maximum(alpha*bn1, bn1)

            fc_2 = tf.layers.dense(lr1, 1024, activation=None, name='fc_2')
            bn2 = tf.layers.batch_normalization(fc_2, training=batch_norm, fused=True)
            lr2 = tf.maximum(alpha*bn2, bn2)
            fc_3 = tf.layers.dense(lr2, 1024, activation=None, name='fc_3')
            bn3 = tf.layers.batch_normalization(fc_3, training=batch_norm, fused=True)
            lr3 = tf.maximum(alpha*bn3, bn3)

            with tf.variable_scope('cam_angles', reuse=reuse):
                fc_4 = tf.layers.dense(lr3, 1024, activation=None, name='fc_4')
                bn4 = tf.layers.batch_normalization(fc_4, training=batch_norm, fused=True)
                lr4 = tf.maximum(alpha*bn4, bn4)
                fc_5 = tf.layers.dense(lr4, 1024, activation=None, name='fc_5')
                bn5 = tf.layers.batch_normalization(fc_5, training=batch_norm, fused=True)
                lr5 = tf.maximum(alpha*bn5, bn5)

                fc_6 = tf.layers.dense(lr5+lr3, 1024, activation=None, name='fc_6')
                bn6 = tf.layers.batch_normalization(fc_6, training=batch_norm, fused=True)
                lr6 = tf.maximum(alpha*bn6, bn6)
                fc_7 = tf.layers.dense(lr6, 1024, activation=None, name='fc_7')
                bn7 = tf.layers.batch_normalization(fc_7, training=batch_norm, fused=True)
                lr7 = tf.maximum(alpha*bn7, bn7)

                fc_embedding = tf.layers.dense(lr7+lr5+lr3, 6, activation=None, name='fc_embedding')
                fc_embedding = tf.nn.tanh(fc_embedding/10.0)
                fc_sin = tf.slice(fc_embedding, [0, 0], [-1, 3], name='layer_sine')
                fc_cos = tf.slice(fc_embedding, [0, 3], [-1, 3], name='layer_cosine')
                deno = tf.sqrt(tf.add(tf.square(fc_sin),tf.square(fc_cos)))
                fc_sin_mod = tf.div(fc_sin,deno)
                fc_cos_mod = tf.div(fc_cos,deno)
                fc_embed_atan2 = tf.atan2(fc_sin_mod, fc_cos_mod, name='layer_atan2')

                fc_cam_params = tf.layers.dense(lr7+lr5+lr3, 2, activation=None, name='fc_cam_params')
 
            with tf.variable_scope('cam_params', reuse=reuse):
                fc_8 = tf.layers.dense(lr3, 1024, activation=None, name='fc_8')
                bn8 = tf.layers.batch_normalization(fc_8, training=batch_norm, fused=True)
                lr8 = tf.maximum(alpha*bn8, bn8)
                fc_9 = tf.layers.dense(lr8, 1024, activation=None, name='fc_9')
                bn9 = tf.layers.batch_normalization(fc_9, training=batch_norm, fused=True)
                lr9 = tf.maximum(alpha*bn9, bn9)

                fc_10 = tf.layers.dense(lr9+lr3, 1024, activation=None, name='fc_10')
                bn10 = tf.layers.batch_normalization(fc_10, training=batch_norm, fused=True)
                lr10 = tf.maximum(alpha*bn10, bn10)
                fc_11 = tf.layers.dense(lr10, 1024, activation=None, name='fc_11')
                bn11 = tf.layers.batch_normalization(fc_11, training=batch_norm, fused=True)
                lr11 = tf.maximum(alpha*bn11, bn11)

                out_pose = tf.layers.dense(lr11+lr9+lr3, (opts['n_joints'] - 4) * 3 + 1, activation=None, name='pose')
                out_pose_reshaped = tf.reshape(out_pose[:, :(opts['n_joints'] - 4) * 3], (-1, opts['n_joints'] - 4, 3))
                print ("out_pose_reshaped ", out_pose_reshaped)

                out_angle = out_pose[:, (opts['n_joints'] - 4) * 3:]

                out_ske = make_skeleton(out_pose_reshaped, out_angle)

            ret_dict['out_cam_angles'] = fc_embed_atan2
            ret_dict['out_cam_params'] = fc_cam_params
            ret_dict['pred_ske'] = out_ske

        return ret_dict


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = tf.layers.conv2d(inputs = input_tensor, filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')
    if batchnorm:
        x = tf.layers.batch_normalization(inputs = x)
    x = tf.nn.relu(x)
    
    # second layer
    x = tf.layers.conv2d(inputs = x, filters = n_filters, kernel_size = (kernel_size, kernel_size), padding = 'same')
    if batchnorm:
        x = tf.layers.batch_normalization(inputs = x)
    x = tf.nn.relu(x)

    return x

def decoder(pose_app_embedding, reuse=False):

    #print pose_app_embedding.shape
    with tf.variable_scope('unified_decoder', reuse=reuse):
        
        upsample6 = tf.layers.conv2d_transpose(inputs = pose_app_embedding, filters = 64, kernel_size = (3,3), strides = (2,2), padding = 'same')
        upsample6 = tf.layers.dropout(inputs = upsample6, rate = 0.1)

        #print ("upsample6: ", upsample6.shape)

        c6 = conv2d_block(input_tensor = upsample6, n_filters = 64)

        #print ("c6: ", c6.shape)

        upsample7 = tf.layers.conv2d_transpose(inputs = c6, filters = 32, kernel_size = (3,3), strides = (2,2), padding = 'same')
        upsample7 = tf.layers.dropout(inputs = upsample7, rate = 0.1)

        #print ("upsample7: ", upsample7.shape)

        c7 = conv2d_block(input_tensor = upsample7, n_filters = 32)

        #print ("c7: ",c7.shape)

        upsample8 = tf.layers.conv2d_transpose(inputs = c7, filters = 16, kernel_size = (3,3), strides = (2,2), padding = 'same')
        upsample8 = tf.layers.dropout(inputs = upsample8, rate = 0.1)

        #print ("upsample8: ", upsample8.shape)

        c8 = conv2d_block(input_tensor = upsample8, n_filters = 16)

        #print ("c8: ",c8.shape)

        upsample9 = tf.layers.conv2d_transpose(inputs = c8, filters = 8, kernel_size = (3,3), strides = (2,2), padding = 'same')
        upsample9 = tf.layers.dropout(inputs = upsample9, rate = 0.1)

        #print ("upsample9: ", upsample9.shape)

        c9 = conv2d_block(input_tensor = upsample9, n_filters = 8)

        #print ("c9: ",c9.shape)

        outputs = tf.layers.conv2d(inputs = c9, filters = 3, kernel_size = (1, 1))

    return outputs


def bg_append_conv(image, background, reuse=False):
    with tf.variable_scope('bg_append', reuse=reuse):
        # background = tf.reshape(background, (tf.shape(background)[0], 128,128,3))
        # print ("image, background = ", image, background)

        batch_norm = True
        with tf.variable_scope('bg_cnn', reuse=reuse):
            bconv1 = tf.layers.conv2d(background, 64, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bbn1 = tf.layers.batch_normalization(bconv1, training=batch_norm, fused=True)
            brelu1 = tf.nn.relu(bbn1)

            bconv2 = tf.layers.conv2d(brelu1, 128, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bbn2 = tf.layers.batch_normalization(bconv2, training=batch_norm, fused=True)
            brelu2 = tf.nn.relu(bbn2)
            
            bconv3 = tf.layers.conv2d(brelu2, 256, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            bbn3 = tf.layers.batch_normalization(bconv3, training=batch_norm, fused=True)
            brelu3 = tf.nn.relu(bbn3)
            
            bconv4 = tf.layers.conv2d(brelu3, 512, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        with tf.variable_scope('img_cnn', reuse=reuse):
            iconv1 = tf.layers.conv2d(image, 64, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            ibn1 = tf.layers.batch_normalization(iconv1, training=batch_norm, fused=True)
            irelu1 = tf.nn.relu(ibn1)

            iconv2 = tf.layers.conv2d(irelu1, 128, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            ibn2 = tf.layers.batch_normalization(iconv2, training=batch_norm, fused=True)
            irelu2 = tf.nn.relu(ibn2)
            
            iconv3 = tf.layers.conv2d(irelu2, 256, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
            ibn3 = tf.layers.batch_normalization(iconv3, training=batch_norm, fused=True)
            irelu3 = tf.nn.relu(ibn3)
            
            iconv4 = tf.layers.conv2d(irelu3, 512, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        concat_image = tf.concat([iconv4, bconv4],3)
        print ("concat_image = ", concat_image)

        cconv1 = tf.layers.conv2d_transpose(concat_image, 512, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        cbn1 = tf.layers.batch_normalization(cconv1, training=batch_norm, fused=True)
        crelu1 = tf.nn.relu(cbn1)

        cconv2 = tf.layers.conv2d_transpose(crelu1, 256, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        cbn2 = tf.layers.batch_normalization(cconv2, training=batch_norm, fused=True)
        crelu2 = tf.nn.relu(cbn2)

        cconv3 = tf.layers.conv2d_transpose(crelu2, 128, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        cbn3 = tf.layers.batch_normalization(cconv3, training=batch_norm, fused=True)
        crelu3 = tf.nn.relu(cbn3)

        cconv4 = tf.layers.conv2d_transpose(crelu3, 64, 3, strides=2, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        cbn4 = tf.layers.batch_normalization(cconv4, training=batch_norm, fused=True)
        crelu4 = tf.nn.relu(cbn4)

        pred_future_image = tf.layers.conv2d(crelu4, 3, 1, strides=1, padding='SAME', 
                                                            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        print ("pred_future_image = ", pred_future_image)

        return pred_future_image


# def bg_append_conv(image, background, reuse=False):
#     with tf.variable_scope('bg_append', reuse=reuse):
#         # background = tf.reshape(background, (tf.shape(background)[0], 128,128,3))
#         # print ("image, background = ", image, background)
#         batch_norm = True
#         with tf.variable_scope('bg_cnn', reuse=reuse):
#             bconv1 = tf.layers.conv2d(background, 64, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#             bbn1 = tf.layers.batch_normalization(bconv1, training=batch_norm, fused=True)
#             brelu1 = tf.nn.relu(bbn1)

#             bconv2 = tf.layers.conv2d(brelu1, 128, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#             bbn2 = tf.layers.batch_normalization(bconv2, training=batch_norm, fused=True)
#             brelu2 = tf.nn.relu(bbn2)
            
#             bconv3 = tf.layers.conv2d(brelu2, 256, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

#         with tf.variable_scope('img_cnn', reuse=reuse):
#             iconv1 = tf.layers.conv2d(image, 64, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#             ibn1 = tf.layers.batch_normalization(iconv1, training=batch_norm, fused=True)
#             irelu1 = tf.nn.relu(ibn1)

#             iconv2 = tf.layers.conv2d(irelu1, 128, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#             ibn2 = tf.layers.batch_normalization(iconv2, training=batch_norm, fused=True)
#             irelu2 = tf.nn.relu(ibn2)
            
#             iconv3 = tf.layers.conv2d(irelu2, 256, 3, strides=2, padding='SAME', 
#                                                             kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

#         concat_image = tf.concat([iconv3, bconv3],3)
#         print ("concat_image = ", concat_image)

#         upsample1 = tf.image.resize_images(concat_image, [56, 56])
#         cconv5 = tf.layers.conv2d(upsample1, 128, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#         cbn5 = tf.layers.batch_normalization(cconv5, training=batch_norm, fused=True)
#         crelu5 = tf.nn.relu(cbn5)
#         cconv6 = tf.layers.conv2d(crelu5, 128, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#         cbn6 = tf.layers.batch_normalization(cconv6, training=batch_norm, fused=True)
#         crelu6 = tf.nn.relu(cbn6)

#         upsample2 = tf.image.resize_images(crelu6, [112, 112])
#         cconv7 = tf.layers.conv2d(upsample2, 64, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#         cbn7 = tf.layers.batch_normalization(cconv7, training=batch_norm, fused=True)
#         crelu7 = tf.nn.relu(cbn7)
#         cconv8 = tf.layers.conv2d(crelu7, 64, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#         cbn8 = tf.layers.batch_normalization(cconv8, training=batch_norm, fused=True)
#         crelu8 = tf.nn.relu(cbn8)

#         upsample3 = tf.image.resize_images(crelu8, [224, 224])
#         cconv9 = tf.layers.conv2d(upsample3, 64, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
#         cbn9 = tf.layers.batch_normalization(cconv9, training=batch_norm, fused=True)
#         crelu9 = tf.nn.relu(cbn9)
#         pred_future_image = tf.layers.conv2d(crelu9, 3, 3, strides=1, padding='SAME', \
#                                                           kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

#         print ("pred_future_image = ", pred_future_image)

#         return pred_future_image


def bg_decoder(bg_embedding, reuse=False):

    with tf.variable_scope('bg_decoder', reuse=reuse):
        batch_norm = True
        
        conv1 = tf.layers.conv2d(bg_embedding, 256, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn1 = tf.layers.batch_normalization(conv1, training=batch_norm, fused=True)
        relu1 = tf.nn.relu(bn1)
        conv2 = tf.layers.conv2d(relu1, 256, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn2 = tf.layers.batch_normalization(conv2, training=batch_norm, fused=True)
        relu2 = tf.nn.relu(bn2)

        upsample1 = tf.image.resize_images(relu2, [32, 32])
        conv3 = tf.layers.conv2d(upsample1, 128, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn3 = tf.layers.batch_normalization(conv3, training=batch_norm, fused=True)
        relu3 = tf.nn.relu(bn3)
        conv4 = tf.layers.conv2d(relu3, 128, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn4 = tf.layers.batch_normalization(conv4, training=batch_norm, fused=True)
        relu4 = tf.nn.relu(bn4)

        upsample2 = tf.image.resize_images(relu4, [64, 64])
        conv5 = tf.layers.conv2d(upsample2, 64, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn5 = tf.layers.batch_normalization(conv5, training=batch_norm, fused=True)
        relu5 = tf.nn.relu(bn5)
        conv6 = tf.layers.conv2d(relu5, 64, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn6 = tf.layers.batch_normalization(conv6, training=batch_norm, fused=True)
        relu6 = tf.nn.relu(bn6)

        upsample3 = tf.image.resize_images(relu6, [128, 128])
        conv7 = tf.layers.conv2d(upsample3, 32, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))
        bn7 = tf.layers.batch_normalization(conv7, training=batch_norm, fused=True)
        relu7 = tf.nn.relu(bn7)
        bg_image = tf.layers.conv2d(relu7, 3, 3, strides=1, padding='SAME', \
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32))

        return bg_image