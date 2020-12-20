# -*- coding:utf-8 -*-
# Author: moranli.aca@gmail.com
# Time: 2020/12/20 4:52 PM
# FileName: eval_RHD.py
# Descriptions:
#


import os.path as osp
import torch
from utils.eval_utils import EvalUtil, calc_auc

def P2W(P, CamI, bl):
    '''
    :param P: Bz,21,3. (u1,v1,d1)
    :param camI: Bz,3,3
    :param bl: Bz,1
    :return: Bz, 21,3. (x0,y0,z0)
    '''
    fx, fy, u0, v0 = CamI[:, 0, 0].unsqueeze(-1), CamI[:, 1, 1].unsqueeze(-1), \
                     CamI[:, 0, 2].unsqueeze(-1), CamI[:, 1, 2].unsqueeze(-1)
    z_cs = P[:, :, -1] * (bl.unsqueeze(-1)) + CamI[:, 2, 2].unsqueeze(-1)

    res = torch.clone(P)

    res[:, :, 0] = z_cs * ((P[:, :, 0] - u0) / fx)
    res[:, :, 1] = z_cs * ((P[:, :, 1] - v0) / fy)
    res[:, :, 2] = z_cs
    return res


def Learnable_Groups_test(model, temperature, test_loader, device=torch.device('cuda')):
    model.eval()
    assert model is not None, "No Net is load, please check the net path!"

    eval_2d, eval_3d = EvalUtil(), EvalUtil()
    pfms_dicts = {'E_mean2D': 0.0, 'E_median2D': 0.0, 'E_auc2d': 0.0, 'E_mean3D': 0.0, 'E_median3D': 0.0,
                  'E_auc3d': 0.0, 'E_auc20-50': 0.0}
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs, kp_uv21_gt, kp_xyz21_norm = data['img_crop'].to(device), \
                                                data['uv_crop'], data['xyz_norm'].to(device)
            uv_pred, depth_pred = model(inputs, temperature)
            kp_uv21_pred, kp_zr_pred_n = uv_pred.detach().cpu(), depth_pred.detach().cpu()
            # evaluate 2D keypoints
            crop_scale = data['crop_scale'].numpy()
            crop_scale = crop_scale.reshape([crop_scale.shape[0], 1, 1])
            eval_2d.feed(kp_uv21_gt.squeeze().numpy() / crop_scale,
                         kp_uv21_pred.squeeze().numpy() / crop_scale)

            cam_mat, norm_scale, kp_xyz21_gt = data['K'], data['norm_scale'], data['xyz']
            pred_uvd = torch.stack([kp_uv21_pred[:, :, 0], kp_uv21_pred[:, :, 1], kp_zr_pred_n], dim=-1)
            pred_xyz = P2W(pred_uvd, cam_mat, norm_scale)
            eval_3d.feed(pred_xyz.squeeze().numpy(), kp_xyz21_gt.squeeze().numpy())

    eval_mean_2d, eval_median_2d, eval_auc_2d, __, __ = eval_2d.get_measures(0.0, 30.0, 20)
    eval_mean_3d, eval_median_3d, eval_auc_3d, pck_curve_all, threshs = eval_3d.get_measures(0.00, 0.050, 20)
    pck_curve_all, threshs = pck_curve_all[8:], threshs[8:] * 1.e3
    auc_subset = calc_auc(threshs, pck_curve_all)
    pfms_dicts['E_mean2D'], pfms_dicts['E_median2D'], pfms_dicts['E_auc2d'], \
    pfms_dicts['E_mean3D'], pfms_dicts['E_median3D'], pfms_dicts['E_auc3d'], pfms_dicts['E_auc20-50'] = \
        round(eval_mean_2d, 3), round(eval_median_2d, 3), round(eval_auc_2d, 3), \
        round(1.0e3 * eval_mean_3d, 4), round(1.0e3 * eval_median_3d, 4), round(eval_auc_3d, 4), round(auc_subset, 3)
    return pfms_dicts

if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader
    from datasets.RHD import RHD_DataReader
    from nets.Net import Learnable_Groups

    parser = argparse.ArgumentParser(description='PyTorch Learnable Grouping Hand Pose Estimation...')
    parser.add_argument('--data_dir', default='/datasets/RHD_published_v2')
    parser.add_argument('--ckpt_dir', default='./net_weights/RHD_final.pth', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    torch.manual_seed(60)


    # path = osp.join('/datasets/RHD_published_v2')  ## path for RHD_published_v2
    assert osp.exists(args.data_dir), 'Please specific the dataset path ...'
    assert osp.exists(args.ckpt_dir), 'Please specific the checkpoint path ...'
    hand_crop, hand_flip, use_wrist, BL, root_id, rotate, uvSigma = True, True, True, 'small', 12, 180, 0.0
    test_set = RHD_DataReader(path=args.data_dir, mode='evaluation', hand_crop=hand_crop, use_wrist_coord=use_wrist, sigma=5, \
                              data_aug=False, uv_sigma=uvSigma, rotate=rotate, BL=BL, root_id=root_id, \
                              right_hand_flip=hand_flip, crop_size_input=256)
    device = torch.device("cuda")
    model = Learnable_Groups([3, True], 256, 21, [32, 32]).to(device)
    ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(ckpt['model_state_dict'])
    temperature = ckpt['Temperature']
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, num_workers=12, drop_last=False, pin_memory=True)
    pfms = Learnable_Groups_test(model, temperature, test_loader)
    print(pfms)


