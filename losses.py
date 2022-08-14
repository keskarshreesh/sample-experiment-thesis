import tensorflow as tf
import numpy as np

# from selfsup.build_vgg16 import build_vgg16
from utils import _exp_running_avg
from config import opts

def l1_loss(prediction, label):
    return tf.reduce_mean(tf.abs(prediction - label))

def masked_l2_loss(prediction, label, mask):
    return tf.reduce_mean(tf.square((prediction - label)*mask))

def l2_loss(prediction, label):
    return tf.reduce_mean(tf.square(prediction - label))

def masked_l1_loss(prediction, label, mask):
    return tf.reduce_sum(tf.abs(tf.multiply(prediction, mask) - tf.multiply(label, mask))) / (tf.reduce_sum(mask)*3)

def add_loss_summary(loss, name, costs_collection, fam='train'):
    tf.summary.scalar(name+'_raw', loss, family=fam)
    tf.add_to_collection(costs_collection, loss)

def _colorization_reconstruction_loss(gt_image, pred_image, loss_mask=None):
    """
    Returns "perceptual" loss between a ground-truth image, and the
    corresponding generated image.
    Uses pre-trained VGG-16 for cacluating the features.

    *NOTE: Important to note that it assumes that the images are float32 tensors
           with values in [0,255], and 3 channels (RGB).

    Follows "Photographic Image Generation".
    """
    with tf.variable_scope('SelfSupReconstructionLoss', reuse=tf.AUTO_REUSE):
        pretrained_file = '/data/vcl/sid/pose_generation/imm/data/models/vgg16.caffemodel.h5' # self._config.perceptual.net_file
        # names = self._config.perceptual.comp
        names = ['input', 'conv1_2','conv2_2','conv3_2','conv4_2','conv5_2']
        ims = tf.concat([gt_image, pred_image], axis=0)
        feats = build_vgg16(ims, pretrained_file=pretrained_file)
        feats = [feats[k] for k in names]
        feat_gt, feat_pred = zip(*[tf.split(f, 2, axis=0) for f in feats])

        ws = [100.0, 1.6, 2.3, 1.8, 2.8, 100.0]
        # f_e = tf.square if self._config.perceptual.l2 else tf.abs
        # f_e = tf.square

        if loss_mask is None:
            loss_mask = lambda x: x

        losses = []
        n_feats = len(feats)
        # n_feats = 3
        # wl = [self._exp_running_avg(losses[k], training_pl, init_val=ws[k], name=names[k]) for k in range(n_feats)]

        for k in range(n_feats):
            l = tf.square(feat_gt[k] - feat_pred[k])
            wl = _exp_running_avg(tf.reduce_mean(loss_mask(l)), init_val=ws[k], name=names[k])
            l /= wl
            l = tf.reduce_mean(loss_mask(l))
            losses.append(l)

        loss = 1000.0*tf.add_n(losses)
    return loss


def loss(ret_dict, costs_collection):

    loss_dict = {}

    # loss_dict['l1_recons_loss'] = l1_loss(ret_dict['recons_img'], ret_dict['tgt_im'])
    # loss_dict['recons_loss'] = _colorization_reconstruction_loss(ret_dict['recons_img'], ret_dict['tgt_im'])
    loss_dict['recons_loss'] = l2_loss(ret_dict['recons_img']*255., ret_dict['label_im']*255.)
    add_loss_summary(loss_dict['recons_loss'],'recons_loss', costs_collection)

    # loss_dict['loss_3d'] = l1_loss(ret_dict['skeleton_3d'], ret_dict['e_syn1_gt_ske_eu'])
    # add_loss_summary(loss_dict['loss_3d'], 'loss_3d', costs_collection)

    loss_dict['loss_2d'] = tf.reduce_mean(ret_dict['loss_flag'])*l2_loss(ret_dict['projs_skeleton_2d'], ret_dict['pose_2d'])*10.
    add_loss_summary(loss_dict['loss_2d'], 'projs_2d_loss', costs_collection)

    # loss_dict['app_loss'] = l1_loss(ret_dict['src_app'], ret_dict['tgt_app'])
    # add_loss_summary(loss_dict['app_loss'], 'app_loss', costs_collection)

    return loss_dict