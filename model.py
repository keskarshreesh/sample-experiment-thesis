# ==========================================================
# Author:  Siddharth Seth
# ==========================================================
"""
Main model file.
"""
from __future__ import division

import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1
import numpy as np

from networks import app_encoder, pose_encoder, bg_append_conv#, resnet50
from transformation import transform_3d_2d, get_skeleton_transform_matrix, root_relative_to_view_norm, make_skeleton
from utils import get_reconstructed_image, gaussian_heatmaps, colorize_landmark_maps
from losses import loss
from config import opts


def add_loss_summary(loss, name, costs_collection, fam='train'):
    tf.summary.scalar(name+'_raw', loss, family=fam)
    tf.add_to_collection(costs_collection, loss)


def model(inputs, flag, training_pl, costs_collection='costs'):
    
    # process input variables
    try:
        ret_dict = {}
        img_siz = opts['image_size']
        n_joints = opts['n_joints']

        src_im = inputs['source_im']
        tgt_im = inputs['future_im']
        back_im = inputs['back_im']
        label_im = inputs['label_im']
        pose_2d = inputs['pose_2d']
        loss_flag = inputs['loss_flag']
        # gauss_hm = inputs['gauss_hm']
        # pose_2d, pose_3d = inputs['pose_2d'], inputs['pose_3d']

        ret_dict['src_im'] = tf.reshape(src_im, (tf.shape(src_im)[0],img_siz,img_siz,3))
        ret_dict['tgt_im'] = tf.reshape(tgt_im, (tf.shape(tgt_im)[0],img_siz,img_siz,3))
        ret_dict['back_im'] = tf.reshape(back_im, (tf.shape(back_im)[0],img_siz,img_siz,3))
        ret_dict['label_im'] = tf.reshape(label_im, (tf.shape(label_im)[0],img_siz,img_siz,3))
        ret_dict['pose_2d'] = tf.reshape(pose_2d, (tf.shape(pose_2d)[0],n_joints,2))
        ret_dict['loss_flag'] = tf.reshape(loss_flag, (tf.shape(loss_flag)[0],1))
        print (ret_dict['tgt_im'])
        print (ret_dict['src_im'])
        print (ret_dict['back_im'])
        print (ret_dict['label_im'])
        print (ret_dict['pose_2d'])
        print (ret_dict['loss_flag'])

        # ret_dict['pose_2d'] = tf.reshape(pose_2d, (tf.shape(pose_2d)[0],n_joints,2))
        # ret_dict['pose_3d'] = tf.reshape(pose_3d, (tf.shape(pose_3d)[0],n_joints,3))

    except Exception as e:
        print ("e in initial ", e)

    ################### get heatmaps from encoder, get 3d points, project to 2d and normalize to image space
    ################### pass in target heatmaps, source app and get the activation image and finally the reconstructed image
    try:
        src_app = app_encoder(ret_dict['src_im'], reuse=tf.AUTO_REUSE)

        # gt_transform_mat= get_skeleton_transform_matrix(ret_dict['pose_3d'])
        # ret_dict['gt_ske_eu'] = root_relative_to_view_norm(ret_dict['pose_3d'], gt_transform_mat)

        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            _, inception_dict = resnet_v1.resnet_v1_50(ret_dict['tgt_im'], is_training=True)

        print (inception_dict.keys())
        # print (inception_dict['resnet_v1_50/block4/unit_3/bottleneck_v1/conv3'])
        # print (inception_dict['resnet_v1_50/block3/unit_6/bottleneck_v1/conv3'])
        # print (inception_dict['resnet_v1_50/block4/unit_2/bottleneck_v1/conv1'])
        # print (inception_dict['resnet_v1_50/block2/unit_3/bottleneck_v1/conv3'])
        # print (inception_dict['resnet_v1_50/block3/unit_4/bottleneck_v1'])
        # print (inception_dict['resnet_v1_50/block3/unit_6/bottleneck_v1/conv2'])
        # print (inception_dict['resnet_v1_50/block4/unit_3/bottleneck_v1'])
        # print (inception_dict['resnet_v1_50/block3/unit_3/bottleneck_v1/conv3'])
        # print (inception_dict['resnet_v1_50/block2/unit_2/bottleneck_v1/conv2'])

        for k in inception_dict.keys():
            print (k, inception_dict[k])

        # resnet_output = resnet50(ret_dict['tgt_im'])['res4f']
        out_dict = pose_encoder(inception_dict['resnet_v1_50/block3/unit_5/bottleneck_v1'], reuse=tf.AUTO_REUSE)
        print ("out_dict ", out_dict)
        
        # ret_dict['f'] = out_dict['f']
        ret_dict['out_cam_angles'] = out_dict['out_cam_angles']
        ret_dict['out_cam_params'] = out_dict['out_cam_params']
        ret_dict['pred_ske'] = out_dict['pred_ske']
        # ret_dict['max_x'] = tf.reduce_max(ret_dict['pose_2d'][:,:,0:1]*112.0+112.0, 1)
        # ret_dict['max_y'] = tf.reduce_max(ret_dict['pose_2d'][:,:,1:2]*112.0+112.0, 1)
        # ret_dict['f'] = tf.concat([ret_dict['max_x'], ret_dict['max_y']], 1)
        
        batch_zeros = tf.zeros_like(out_dict['out_cam_angles'][:, 0:1])
        trans = 20.0
        ret_dict['trans'] = tf.reshape(tf.stack([batch_zeros, batch_zeros+trans, batch_zeros], 1), (-1,3,1))
        out_dict = transform_3d_2d(ret_dict)
        ret_dict['rot_ske'] = out_dict['rot_ske']
        ret_dict['projs_skeleton_2d'] = out_dict['projs_skeleton_2d']
        ret_dict['unscaled_projs_skeleton_2d'] = out_dict['unscaled_projs_skeleton_2d']
        ret_dict['focal_length'] = out_dict['focal_length']

        projs_2d_pose, projs_2d_pose_vis = gaussian_heatmaps(ret_dict['projs_skeleton_2d'])
        print ("projs_2d_pose_vis ", projs_2d_pose_vis)
        print ("projs_2d_pose ", projs_2d_pose)
        pose_embed_tgt = colorize_landmark_maps(projs_2d_pose_vis[1])

        pose_app_img, pose_app_vis = get_reconstructed_image(projs_2d_pose[0], src_app, ret_dict['back_im'], reuse=tf.AUTO_REUSE)

        ret_dict['recons_img'] = tf.clip_by_value(pose_app_img, 0., 1.)
        ret_dict['pose_app_vis'] = pose_app_vis
        ret_dict['pose_embed_tgt'] = pose_embed_tgt

    except Exception as e:
        print ("e in encoder setup ", e)

    ############ calculate losses
    try:
        loss_dict = loss(ret_dict, costs_collection)

    except Exception as e:
        print ("e in losses ", e)

    # image_summary('results', combined_results)
    return loss_dict, ret_dict
