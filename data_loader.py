import os.path as osp
import os
import cv2
import numpy as np
import random
import copy
import scipy.io as sio
import tensorflow as tf
import glob as glob
import json
from natsort import natsorted
import math
import imutils


# human36_datapath = '/data/val/shreesh/shreesh2.7/center_crops/'
# human36_datapath_dirs = os.listdir(human36_datapath)
# print ("human36_datapath_dirs ", len(human36_datapath_dirs))
# human36_datapath_dirs.sort()
# human36bg_datapath = '/data/vcl/sid/human36_parsed/parsed_data/bkg_image/'
# human36bg_datapath_dirs = os.listdir(human36bg_datapath)
# human36bg_datapath_dirs.sort()


def read_dummy_list():

    dummy = []
    f = open("dummy_list.txt", "r")
    for x in f:
        dummy.append(x)
    
    # dummy = [d.split('\n') for d in dummy]
    # dummy = [d[0] for d in dummy if d != '\n']

    print ("(dummy) ", dummy[:50])
    print ("type(dummy) ", type(dummy))
    return dummy, copy.copy(dummy)#[:10000]


def get_crop_params(target_file_name):
    # crop_params_ = 12 i.e. ends at 11 index
    try:
        crop_params = target_file_name[12:]
        params = crop_params.split('_')
        # print ("target_file_name, params ", target_file_name, params)
        # y_min, y_max, x_min, x_max = int(params[0]), int(params[1]), int(params[2]), int(params[3])
        x_min, x_max, y_min, y_max = int(params[0]), int(params[1]), int(params[2]), int(params[3])
        frame_no = int(params[4].split('.')[0])
        # print ("target_file_name ", target_file_name)
        # print ("y_min, y_max, x_min, x_max ", y_min, y_max, x_min, x_max)
    except Exception as e:
        print ("e in get_crop_params ", e)
    return y_min, y_max, x_min, x_max# , frame_no
# crop_params_0_368_0_368_0000.png


def get_cropped_poses(img_orig_shape, mat_2d, img_size):

    m_2d = mat_2d.copy()
    
    x_min = int(np.min(mat_2d[:, 0])) -10
    x_max = int(np.max(mat_2d[:, 0])) + 10
    y_min = int(np.min(mat_2d[:, 1])) - 10
    y_max = int(np.max(mat_2d[:, 1])) + 10
    
    y_dist = y_max-y_min
    
    x_dist = x_max-x_min
    
    if y_dist > x_dist:
        dist = (y_dist - x_dist)/2.0
        dist = int(dist)
        if (x_min-dist) < 0:
            x_min_full = 0
            x_max_full = y_dist
        elif (x_max + dist)>(img_orig_shape[1]-1):
                x_max_full = (img_orig_shape[1]-1)
                x_min_full = x_max_full - y_dist
        else:
            x_min_full = x_min - dist
            x_max_full = min(img_orig_shape[1]-1, x_min_full + y_dist)
        ratio = float(img_size)/(x_max_full - x_min_full)
        m_2d[:, 1] -= y_min
        m_2d[:, 0] -= x_min_full
    else:
        dist = (x_dist - y_dist)/2.0
        dist = int(dist)
        if (y_min-dist) < 0:
            y_min_full = 0
            y_max_full = x_dist
        elif (y_max + dist)>(img_orig_shape[0]-1):
                y_max_full = (img_orig_shape[0]-1)
                y_min_full = y_max_full - x_dist
        else:
            y_min_full = y_min - dist
            y_max_full = min(img_orig_shape[0]-1, y_min_full + x_dist)
        
        img_center_crop_params = [y_min_full, y_max_full, x_min, x_max]
        ratio = float(img_size)/(y_max_full - y_min_full)
        m_2d[:, 1] -= y_min_full
        m_2d[:, 0] -= x_min
        
    m_2d *= ratio
    return m_2d

def perform_flip(points):
    points = np.stack([points[:,0] + 2*(112 - points[:,0]), points[:,1]], axis=1)
    points_flipped = np.stack([points[0], points[1], points[5], points[6], points[7], \
                                     points[2], points[3], points[4], points[8], points[13], \
                                     points[14], points[15], points[16], points[9], points[10], points[11], points[12]], axis=0)
    return points_flipped

def perform_tilt(points, tilt_angle):
    gauss_mu_x_tilt = ( (points[:,1] - 112.0) * np.cos(tilt_angle) - (points[:,0] - 112.0) * np.sin(tilt_angle) ) + 112.0
    gauss_mu_y_tilt = ( (points[:,1] - 112.0) * np.sin(tilt_angle) + (points[:,0] - 112.0) * np.cos(tilt_angle) ) + 112.0

    # print ("gauss_mu_x_tilt ", gauss_mu_x_tilt)
    # print ("gauss_mu_y_tilt ", gauss_mu_y_tilt)
    return np.stack([gauss_mu_y_tilt, gauss_mu_x_tilt], axis=1)

mads_datapath = '/data/vcl/sid/mads_parsed/parsed_data/center_crops'
mads_datapath_dirs = os.listdir(mads_datapath)
print ("mads_datapath_dirs ", len(mads_datapath_dirs))
mads_datapath_dirs.sort()
madsbg_datapath = '/data/vcl/sid/mads_parsed/parsed_data/bkg_image'
madsbg_datapath_dirs = os.listdir(madsbg_datapath)
madsbg_datapath_dirs.sort()
madscrop_datapath = '/data/vcl/sid/mads_parsed/parsed_data/cropped_frames'
mads_posepath_2d = '/data/vcl/sid/mads_parsed/parsed_data/poses_2d'

sup_folders = ['HipHop_HipHop1_C0', 'HipHop_HipHop6_C1', 'Jazz_Jazz1_C1', 'Jazz_Jazz6_C2', 'Sports_Badminton_C2', 'Sports_Volleyball_Left']

# read unsupervised data
def _read_py_function1(dummy):

    while True:
        try:

            datapath = mads_datapath
            bg_datapath = madsbg_datapath
            folder_idx = np.random.randint(len(mads_datapath_dirs))
            folder = mads_datapath_dirs[folder_idx]

            if folder.split('_video')[0] in str(sup_folders):
                continue 

            subject_id = folder[:folder.find('_')]    # for randomly selecting another folder for taking out future frame

            num_frames = len(os.listdir(os.path.join(mads_datapath, folder)))
            frame_idx = np.random.randint(num_frames-10)    # select a random frame from this folder

            target_folder_idx = folder_idx + 2
            if target_folder_idx >= len(mads_datapath_dirs):
                target_folder_idx = folder_idx - 2
            target_subject_id = mads_datapath_dirs[target_folder_idx][:mads_datapath_dirs[target_folder_idx].find('_')]
            if target_subject_id == subject_id:
              target_folder = mads_datapath_dirs[target_folder_idx]
              num_frames = len(os.listdir(os.path.join(mads_datapath, target_folder)))
              target_frame_idx = np.random.randint(num_frames-10)

            else:
              target_folder_idx = folder_idx - 2
              target_folder = mads_datapath_dirs[target_folder_idx]
              num_frames = len(os.listdir(os.path.join(mads_datapath, target_folder)))
              target_frame_idx = np.random.randint(num_frames-10)


            file_name = os.listdir(os.path.join(datapath, folder))[frame_idx]
            source_image = cv2.imread(os.path.join(os.path.join(datapath, folder), file_name))

            target_file_name = os.listdir(os.path.join(datapath, target_folder))[target_frame_idx]
            target_image = cv2.imread(os.path.join(os.path.join(datapath, target_folder), target_file_name))

            x_min, x_max, y_min, y_max = get_crop_params(target_file_name)
            pos = target_folder.find('_video')
            bg_name = target_folder[:pos]

            background_image = cv2.imread(os.path.join(bg_datapath, bg_name+'.jpg'))
            background_image = background_image[y_min:y_max, x_min:x_max, :]

            target_index_actual = int(target_file_name.split('_')[-1].split('.')[0]) + int(target_folder.split('_')[-3])
            pose_2d = sio.loadmat(os.path.join(mads_posepath_2d, bg_name+'.mat'))['pose_2d'][target_index_actual]
            pose_2d = get_cropped_poses([384, 512], pose_2d, 224)

            label_image = target_image.copy()
            source_image = cv2.resize(source_image, (224, 224))
            target_image = cv2.resize(target_image, (224, 224)) - np.array([103.939, 116.779, 123.68])
            background_image = cv2.resize(background_image, (224, 224))
            label_image = cv2.resize(label_image, (224, 224))

            # fg_mask = (1. - (label_image == background_image))*99. + 1.

            source_image = source_image[:,:,::-1].astype(np.float32) /255.0
            target_image = target_image[:,:,::-1].astype(np.float32) /255.0
            background_image = background_image[:,:,::-1].astype(np.float32) /255.0
            label_image = label_image[:,:,::-1].astype(np.float32) /255.0
            break
        except Exception as e:
            print ("e py1", e)
            continue

    return source_image, target_image, label_image, background_image, pose_2d.astype(np.float32), np.zeros((1,1)).astype(np.float32) #, label


# read supervised data
def _read_py_function2(dummy):

    while True:
        try:

            datapath = mads_datapath
            bg_datapath = madsbg_datapath
            folder_idx = np.random.randint(len(mads_datapath_dirs))
            folder = mads_datapath_dirs[folder_idx]

            if folder.split('_video')[0] not in str(sup_folders):
                continue 

            subject_id = folder[:folder.find('_')]    # for randomly selecting another folder for taking out future frame

            num_frames = len(os.listdir(os.path.join(mads_datapath, folder)))
            frame_idx = np.random.randint(num_frames-10)    # select a random frame from this folder

            target_folder_idx = folder_idx + 2
            if target_folder_idx >= len(mads_datapath_dirs):
                target_folder_idx = folder_idx - 2
            target_subject_id = mads_datapath_dirs[target_folder_idx][:mads_datapath_dirs[target_folder_idx].find('_')]
            if target_subject_id == subject_id:
              target_folder = mads_datapath_dirs[target_folder_idx]
              num_frames = len(os.listdir(os.path.join(mads_datapath, target_folder)))
              target_frame_idx = np.random.randint(num_frames-10)

            else:
              target_folder_idx = folder_idx - 2
              target_folder = mads_datapath_dirs[target_folder_idx]
              num_frames = len(os.listdir(os.path.join(mads_datapath, target_folder)))
              target_frame_idx = np.random.randint(num_frames-10)


            file_name = os.listdir(os.path.join(datapath, folder))[frame_idx]
            source_image = cv2.imread(os.path.join(os.path.join(datapath, folder), file_name))

            target_file_name = os.listdir(os.path.join(datapath, target_folder))[target_frame_idx]
            target_image = cv2.imread(os.path.join(os.path.join(datapath, target_folder), target_file_name))

            x_min, x_max, y_min, y_max = get_crop_params(target_file_name)
            pos = target_folder.find('_video')
            bg_name = target_folder[:pos]

            background_image = cv2.imread(os.path.join(bg_datapath, bg_name+'.jpg'))
            background_image = background_image[y_min:y_max, x_min:x_max, :]

            target_index_actual = int(target_file_name.split('_')[-1].split('.')[0]) + int(target_folder.split('_')[-3])
            pose_2d = sio.loadmat(os.path.join(mads_posepath_2d, bg_name+'.mat'))['pose_2d'][target_index_actual]
            pose_2d = get_cropped_poses([384, 512], pose_2d, 224)

            label_image = target_image.copy()
            source_image = cv2.resize(source_image, (224, 224))
            target_image = cv2.resize(target_image, (224, 224)) - np.array([103.939, 116.779, 123.68])
            background_image = cv2.resize(background_image, (224, 224))
            label_image = cv2.resize(label_image, (224, 224))

            if np.random.choice(2) == 1:
                rand_angles = np.random.rand(1)
                pose_tilt_val = rand_angles * 0.17
                rand_angles_with_sign = (rand_angles * 2) - 1.0
                rand_angles_with_sign = rand_angles_with_sign / np.abs(rand_angles_with_sign)
                pose_tilt_val = pose_tilt_val * rand_angles_with_sign
                if pose_tilt_val < 0:
                  pose_tilt_val -= 0.35
                else:
                  pose_tilt_val += 0.35
                # try:
                #     pose_tilt_val = np.reshape(pose_tilt_val, (1, 1))
                # except Exception as e:
                #     print ("e in reshape ", e)
                # try:
                pose_2d = perform_tilt(pose_2d, pose_tilt_val)
                # print ("pose_2d.shape ", pose_2d.shape)
                # assert pose_2d.shape[0] == 17
                # except Exception as e:
                #     print ("pose_2d.shape[0] ", pose_2d.shape)
                #     print ("e in tilt ", e)
                # try:
                target_image = imutils.rotate(target_image, pose_tilt_val*180./math.pi)
                # except Exception as e:
                # print ("e in rotate tgt ", e)
                # try:
                background_image = imutils.rotate(background_image, pose_tilt_val*180./math.pi)
                # except Exception as e:
                #     print ("pose_2d.shape ", pose_2d.shape)
                #     print ("e in rotate bg ", e)
                label_image = imutils.rotate(label_image, pose_tilt_val*180./math.pi)
                
            elif np.random.choice(2) == 1:
                # try:
                pose_2d = perform_flip(pose_2d)
                # assert pose_2d.shape[0] == 17
                target_image = cv2.flip(target_image, 1)
                background_image = cv2.flip(background_image, 1)
                label_image = cv2.flip(label_image, 1)
                # except Exception as e:
                #     print ("pose_2d.shape ", pose_2d.shape)
                #     print ("e in flipping ", e)
            # fg_mask = (1. - (label_image == background_image))*99. + 1.

            source_image = source_image[:,:,::-1].astype(np.float32) /255.0
            target_image = target_image[:,:,::-1].astype(np.float32) /255.0
            background_image = background_image[:,:,::-1].astype(np.float32) /255.0
            label_image = label_image[:,:,::-1].astype(np.float32) /255.0
            break
        except Exception as e:
            print ("e py2 ", e)
            continue

    return source_image, target_image, label_image, background_image, pose_2d.astype(np.float32), np.ones((1,1)).astype(np.float32) #, label