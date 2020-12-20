# -*- coding:utf-8 -*-
# FileName: RHD.py
# Descriptions: dataset processing. modified from:
# < https://github.com/lmb-freiburg/hand3d >
# Data Augmentation Configs are similar to < Aligning Latent Spaces for 3D Hand Pose Estimation. ICCV2019 >

from __future__ import print_function
import cv2, pickle
import os.path as osp
import numpy as np
from skimage import io, transform
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as TF

__all__ = ["RHD_DataReader"]


def get_hand_flag(hand_parts):
    one_map, zero_map = np.ones_like(hand_parts), np.ones_like(hand_parts)
    cond_l = np.logical_and(np.greater(hand_parts, one_map), np.less(hand_parts, one_map * 18))  # left hands mask
    cond_r = np.greater(hand_parts, one_map * 17)  # right hands mask
    choose_hand_bool = np.greater(np.sum(np.sum(cond_l)), np.sum(np.sum(cond_r)))  # True for choose left

    return choose_hand_bool


def coord_normalize(coord, root_id, BL):
    '''
    :param coord: (21,3) coordinates
    root_id: 12 for gt_12
    BL: small or  large
    :return:
    '''
    coord_rel = coord - coord[root_id, :]
    assert BL in ['large', 'small'], 'bone length specific error...'
    if BL == 'large':
        bone_length = np.sqrt(np.sum(np.square(coord_rel[12, :] - coord_rel[0, :])))
    elif BL == 'small':
        bone_length = np.sqrt(np.sum(np.square(coord_rel[12, :] - coord_rel[11, :])))

    coord_norm = coord_rel / bone_length  # normalized by length of 12->11(12->0)

    return coord_norm, bone_length


class RHD_DataReader(Dataset):
    """
        read RHD dataset as required
    """

    def __init__(self, mode=None, path=None, hand_crop=True,
                 use_wrist_coord=True, sigma=25.0,
                 data_aug=False, rotate=180, uv_sigma=2.0, root_id=12, BL='small',
                 scoremap_dropout=False, crop_size_input=256, right_hand_flip=False):
        assert (mode == 'training' or mode == 'evaluation'), RuntimeError('Unrecognized mode')
        self.path, self.mode = path, mode
        self.right_hand_flip = right_hand_flip
        path_to_anno = osp.join(path, f'{self.mode}/anno_{self.mode}.pickle')
        assert osp.exists(path_to_anno), f'annotation file: {path_to_anno} not exist...'
        self.hand_crop = hand_crop

        self.data_aug, self.rotate = data_aug, rotate ## true for using coord noise, rotate, crop_scale noise, crop_offset_noise, and color jetter
        self.coord_uv_noise_sigma = uv_sigma  # 2.5
        self.crop_offset_noise_sigma = 1.0
        self.BL, self.root_id = BL, root_id
        self.translate_range = 20

        self.crop_size, self.use_wrist_coord = crop_size_input, use_wrist_coord


        self.hand_side = 0
        self.img_w, self.img_h = 320, 320


        #         assert self.root_id == 12 and self.rot == 180 and self.translate_range == 20 and self.crop_size == 256 and self.use_wrist_coord, 'Check the config to be the same as Align_ICCV2019'

        with open(path_to_anno, 'rb') as fi:
            self.anno_all = pickle.load(fi)

    def __len__(self):
        return len(self.anno_all.items())

    def __getitem__(self, idx):

        """READ DATA"""
        # 1. read image
        imgname = osp.join(self.path, self.mode, 'color', '%.5d.png' % idx)
        assert osp.exists(imgname), RuntimeError(f'Can not read image! {imgname}')
        self.image = io.imread(imgname)
        # 2. read mask
        assert osp.exists(imgname.replace('color', 'mask')), RuntimeError('Can not read mask!')
        hand_parts = io.imread(imgname.replace('color', 'mask'))  ## [320,320]

        # 3. read annotation
        # read keypoint uv
        kp_coord_uv_, kp_visible_ = self.anno_all[idx]['uv_vis'][:, :2], self.anno_all[idx]['uv_vis'][:, 2]
        kp_coord_xyz_ = self.anno_all[idx]['xyz']
        camera_intrinsic_matrix = self.anno_all[idx]['K'].reshape([3, 3])
        self.fx, self.fy = camera_intrinsic_matrix[0, 0], camera_intrinsic_matrix[0, 0]
        self.u0, self.v0 = camera_intrinsic_matrix[0, -1], camera_intrinsic_matrix[0, -1]

        if not self.use_wrist_coord:
            kp_coord_uv_[0, :] = 0.5 * (kp_coord_uv_[0, :] + kp_coord_uv_[12, :])
            kp_coord_uv_[21, :] = 0.5 * (kp_coord_uv_[21, :] + kp_coord_uv_[33, :])

            kp_visible_[0] = np.logical_or(kp_visible_[0], kp_visible_[12]).astype(float)
            kp_visible_[21] = np.logical_or(kp_visible_[21], kp_visible_[33]).astype(float)

            kp_coord_xyz_[0, :] = 0.5 * (kp_coord_xyz_[0, :] + kp_coord_xyz_[12, :])
            kp_coord_xyz_[21, :] = 0.5 * (kp_coord_xyz_[21, :] + kp_coord_xyz_[33, :])

        if self.data_aug and self.coord_uv_noise_sigma > 0:
            noise = np.random.normal(loc=0.0, scale=self.coord_uv_noise_sigma, size=[42, 2])  ### adding uv noise
            kp_coord_uv_ += noise

        """GET THE SUBSET of 21 keypoints"""

        choose_hand_bool = get_hand_flag(hand_parts)

        cond_left = np.logical_and(np.ones((21, 3)).astype(bool), choose_hand_bool)
        kp_coord_xyz = np.where(cond_left, kp_coord_xyz_[:21, :],
                                kp_coord_xyz_[-21:, :])  # if left hands pixel more,kp is left hands coords

        self.hand_side = np.where(choose_hand_bool, 0, 1)  # find left or right hands

        # Set of 21 for visibility

        self.kp_vis21 = np.where(cond_left[:, 0], kp_visible_[:21], kp_visible_[-21:])  # save the domain hands vis21

        # Set of 21 for UV coordinates
        kp_coord_uv = np.where(cond_left[:, :2], kp_coord_uv_[:21, :],
                               kp_coord_uv_[-21:, :])  # save the domain hands uv21

        # Adding random rotate if data_aug
        angle = np.random.rand() * 2 * self.rotate - self.rotate if self.data_aug else 0.0

        self.image = np.array(TF.rotate(TF.to_pil_image(self.image), angle))
        cos_, sin_ = np.cos(-(angle / 180.0) * np.pi), np.sin(-(angle / 180.0) * np.pi)
        rotate_M0 = np.array([[cos_, sin_], [-sin_, cos_]])
        rotate_b0 = np.array(
            [(1 - cos_) * self.img_h / 2 + sin_ * self.img_w / 2, (1 - cos_) * self.img_w / 2 - sin_ * self.img_h / 2])

        self.kp_uv21_rotate = kp_coord_uv.dot(rotate_M0) + rotate_b0

        # make coords relative to root joint

        # above all is about the entire picture 320*320 coord and mask .thus the pixel of hands is few
        # next will find the cropped hands with the corresponding uv21,xyz kepoint
        """hand crop"""
        if self.hand_crop:
            kp_coord_hw = []
            for i in range(len(self.kp_vis21)):
                if self.kp_vis21[i] == 1:
                    kp_coord_hw.append([self.kp_uv21_rotate[i, 1], self.kp_uv21_rotate[i, 0]])
            kp_coord_hw = np.array(kp_coord_hw)  # find tightest bbox for hands

            if len(kp_coord_hw) > 0:
                min_coord = np.maximum(np.min(kp_coord_hw, 0), 0.0)  # find min (y,x) of all(y,x)
                _min_coord = np.maximum(kp_coord_hw.min(axis=0), 0.0)
                
                max_coord = np.minimum(np.max(kp_coord_hw, 0),
                                       self.img_w)  # find max (y.x)#this process can not include entire hand
                _max_coord = np.minimum(kp_coord_hw.max(axis=0), self.img_w)
                assert np.sum(_min_coord - min_coord) == 0 and np.sum(_max_coord - max_coord) == 0, 'sum error coord'
#                 print('max_coord', max_coord)
            else:
                kp_hw21 = [self.kp_uv21_rotate[:, 1], self.kp_uv21_rotate[:, 0]]
                kp_hw21_t = np.reshape(kp_hw21, [2, 21]).T
                min_coord = np.maximum(np.min(kp_hw21_t, 0), 0.0)
                max_coord = np.minimum(np.max(kp_hw21_t, 0), self.img_w)
                
            crop_center = self.kp_uv21_rotate[12, ::-1]             
            crop_size = np.max(2 * np.maximum(max_coord - crop_center, crop_center - min_coord))
            crop_size = 1.2 * np.clip(crop_size, 30, 500)
            crop_center_tmp = 0.5 * (min_coord + max_coord)  # using this will make the whole hand in the center of picture

            m_coord = np.tile(crop_center_tmp.reshape(-1, 1), [1, 2]) + np.array(
                [-crop_size / 2, crop_size / 2]).reshape(2, )
            if self.data_aug:
                crop_scale_noise = np.random.random() * 0.2 + 1  ### adding scale noise to [0.8, 1.1]
                noise = np.random.normal(loc=0.0, scale=self.crop_offset_noise_sigma, size=[2])

                crop_size_new = crop_size * crop_scale_noise

                m_coord += np.array([-(crop_size_new - crop_size) / 2, (crop_size_new - crop_size) / 2]).reshape(2, )
                m_coord += noise.reshape(2, 1)
                m_coord += np.random.randint(-self.translate_range, self.translate_range, 1)

                self.transform = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize((self.crop_size)),
                                                     transforms.ColorJitter(hue=[-0.1, 0.1]),
                                                     transforms.ToTensor(),
                                                     ])

            else:
                self.transform = transforms.Compose([transforms.ToPILImage(),
                                                     transforms.Resize((self.crop_size, self.crop_size)),
                                                     transforms.ToTensor(),
                                                     ])

            min_coord = np.clip(m_coord[:, 0], 10.0, self.img_w - 10.0)
            max_coord = np.clip(m_coord[:, 1], 10.0, self.img_w - 10.0)

            crop_size = np.max(max_coord - min_coord)
            # crop_size_best = np.minimum(np.maximum(crop_size, 30), 280)
            crop_size_best = np.clip(crop_size, 30, 280)

            bx = int(np.max([int(min_coord[1]) + int(crop_size_best) - self.img_w + 10, 0]))
            by = int(np.max([int(min_coord[0]) + int(crop_size_best) - self.img_w + 10, 0]))
            y1, x1 = int(min_coord[0]) - by, int(min_coord[1]) - bx
            x2, y2 = x1 + int(crop_size_best), y1 + int(crop_size_best)

            # Crop image
            img_crop = self.image[int(y1):int(y2), int(x1):int(x2), :]

            self.img_crop = self.transform(img_crop)
            self.crop_scale = self.crop_size / crop_size_best
            self.kp_uv21_crop = self.crop_scale * (self.kp_uv21_rotate - [x1, y1])
            self.cam_mat_crop = self.crop_scale * np.array([[self.fx, 0, (self.u0 - x1)],
                                                            [0, self.fy, (self.u0 - y1)],
                                                            [0, 0, kp_coord_xyz[self.root_id, -1] / self.crop_scale]])
            cam_mat_uv2xyz = np.array([[1 / self.cam_mat_crop[0, 0], 0, 0],
                                       [0, 1 / self.cam_mat_crop[1, 1], 0],
                                       [-self.cam_mat_crop[0, 2] / self.cam_mat_crop[0, 0],
                                        -self.cam_mat_crop[1, 2] / self.cam_mat_crop[1, 1], 1]])

        if self.right_hand_flip:
            if self.hand_side == 0:
                # flip left hands to right hands
                self.img_crop = cv2.flip(self.img_crop.permute(1, 2, 0).numpy(), 1)
                self.img_crop = torch.from_numpy(self.img_crop).float().permute(2, 0, 1)

                self.kp_uv21_crop = np.tile(np.array([-1, 1]), [21, 1]) * (
                        self.kp_uv21_crop + np.tile(np.array([1 - self.crop_size, 0]), [21, 1]))
            k_uv1 = np.concatenate([self.kp_uv21_crop, np.ones([self.kp_uv21_crop.shape[0], 1])], axis=-1)
            self.kp_xyz_rotate = kp_coord_xyz[:, 2].reshape(-1, 1) * k_uv1.dot(cam_mat_uv2xyz)

        self.kp_xyz21_norm, self.kp_norm_scale = coord_normalize(self.kp_xyz_rotate, root_id=self.root_id, BL=self.BL)

        self.img_crop = self.img_crop.numpy().astype(np.float32)

        sample = {'img_crop': self.img_crop,
                  'crop_scale': self.crop_scale,
                  'uv_crop': self.kp_uv21_crop,
                  'vis21': self.kp_vis21,
                  'xyz': self.kp_xyz_rotate,
                  'xyz_norm': self.kp_xyz21_norm,
                  'norm_scale': self.kp_norm_scale,
                  'K': self.cam_mat_crop
                  }

        return sample