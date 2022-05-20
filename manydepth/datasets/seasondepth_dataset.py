# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil
from PIL import Image

import cv2
from .mono_dataset import MonoDataset
from .mono_dataset_seasondepth import MonoDatasetSeason

class SeasonDepthDataset(MonoDatasetSeason):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(SeasonDepthDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        # To normalize you need to scale the first row by 1 / image_width and the second row
        # by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        # If your principal point is far from the center you might need to disable the horizontal
        # flip augmentation.

        # c0
        
        self.Kc0 = np.array([[0.848626, 0, 0.513616, 0],
                           [0, 1.127686, 0.546930, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        
        # c1
        self.Kc1 = np.array([[0.852913, 0, 0.516918, 0],
                           [0, 1.141262, 0.517282, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        

        self.full_res_shape = (1024, 768)
        self.side_map = {"c0": 1, "c1": 2}

    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split('/')
    
        side = line[3]
        # img_path_line = line[-1].split('_')
        # frame_id = int(img_path_line[1])
        
        # img_path_front = os.path.join(self.data_path,line[0],line[1],line[2],line[3],line[4])
        # img_path = os.path.join(self.data_path,self.filenames[index])
        
        return side
    
    def check_depth(self):
        line = self.filenames[0].split("/")
        folder = self.filenames[0]
        frame_index = ''
        side = ''

        if line[4] == 'images':  # train_set
            name = line[5].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth","train", line[1], line[2], line[3], 'depth_map', name[0] + ".png")
        elif line[1] == 'images':  # val_set
            name = line[4].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth","val", "depth", line[2], line[3], name[0] + ".png")
        velo_filename = '/' + velo_filename
        return os.path.isfile(velo_filename)

    def get_color(self,index, i,do_flip):
    
        if index + i < 0 or index + i == len(self.filenames):
            return Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
        else:
            line_current = self.filenames[index].split('/')
            line_last = self.filenames[index+i].split('/')
            img_path_line_c = line_current[-1].split('_')
            frame_id_c = int(img_path_line_c[1])
            img_path_line_l = line_last[-1].split('_')
            frame_id_l = int(img_path_line_l[1])
            if frame_id_c + i != frame_id_l:
                return Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                
            img_path = os.path.join(self.data_path, self.filenames[index + i])
            color = self.loader(img_path)
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            return color

            side = line[3]

            img_path_back = img_path_line[0] + "_{:05d}".format(frame_id + i) + "_{}_".format(side) + img_path_line[-1]
            img_path = os.path.join(self.data_path)
            for words in line[:-1]:
                img_path = os.path.join(words)
    
            img_path = os.path.join(self.data_path, self.filenames[index + i])







    
            line = self.filenames[index].split('/')
        try:
            line = self.filenames[index].split('/')
        
            # side = line[3]
            # img_path_line = line[-1].split('_')
            # frame_id = int(img_path_line[1])
            # img_path_back = img_path_line[0] + "_{:05d}".format(frame_id + i) + "_{}_".format(side) + img_path_line[-1]
            # img_path = os.path.join(self.data_path)
            # for words in line[:-1]:
            #     img_path = os.path.join(words)

            line = self.filenames[index].split('/')

            side = line[3]
            # img_path_line = line[-1].split('_')
            # frame_id = int(img_path_line[1])

            # img_path_front = os.path.join(self.data_path,line[0],line[1],line[2],line[3],line[4])
            # img_path = os.path.join(self.data_path,self.filenames[index])


            if index + i < 0 or index + i == len(self.filenames):
                return Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
            else:
                img_path = os.path.join(self.data_path, self.filenames[index + i])
            color = self.loader(img_path)
    
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
        except:
            print(self.filenames[index])
            print(index)
            
        return color


class SDDataset(SeasonDepthDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(SDDataset, self).__init__(*args, **kwargs)


    def readlines(self,filename):
        """Read all the lines in a text file and return as a list
        """
        with open(filename, 'r') as f:
            lines = f.read().splitlines()
        return lines
    
    def get_image_path(self, folder, frame_index, side):
        line = self.filenames[0].split("/")

        if line[4] == 'images':  # train_set
            img_pth = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "train", line[1], line[2], line[3], line[4],line[5])
        elif line[1] == 'images':  # val_set
            img_pth = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "val", line[1], line[2], line[3], line[4])
        img_pth = '/' + img_pth
        return img_pth

    def get_depth(self, folder, frame_index, side, do_flip):
        line = folder.split("/")

        if line[4] == 'images':  # train_set
            name = line[5].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "train", line[1], line[2], line[3], 'depth_map',
                name[0] + ".png")
        elif line[1] == 'images':  # val_set
            name = line[4].split(".")
            velo_filename = os.path.join(
                "data0", "dataset", "SJTU", "SeasonDepth", "val", "depth", line[2], line[3], name[0] + ".png")
        velo_filename = '/' + velo_filename
        depth_gt = cv2.imread(velo_filename, -1)
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt