# -*- coding:utf-8 -*-
# Author: moranli.aca@gmail.com
# Time: 2021/1/21 15:28
# FileName: plot_AUC_tools.py
# Descriptions: for comparison between SOTA methods on RHD/STB/Dexter+Object
#


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def get_color_list():
    color_list = ['steelblue', 'orange', 'green', 'mediumorchid', 'violet', 'brown', 'lightsteelblue', 'aquamarine', 'palegoldenrod']
    return color_list

def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm

def plt_STB(save_dir="STB_OursOverallG3.pdf"):


    ours_thresh_ = np.array([21.052631578947366, 23.68421052631579, 26.31578947368421, 28.94736842105263,
                             31.578947368421055, 34.21052631578947, 36.84210526315789, 39.473684210526315,
                             42.10526315789473, 44.73684210526316, 47.36842105263158, 50.0])
    ours_acc = np.array([0.9809047619047617,0.9885079365079363,0.992142857142857,
                         0.9946031746031748,0.9963333333333333,0.99715873015873,
                         0.9978571428571428,0.9983809523809523,0.9987142857142857,
                         0.9989523809523808,0.9991269841269841,0.9992539682539684])
    ours_auc = 0.9956233766233765

    #  Public AUC data
    align_latentSpace_acc0 = np.array([0., 0.19173016, 0.44801587, 0.65598413, 0.79346032, 0.8781746,
                                       0.92895238, 0.95992063, 0.97728571, 0.98715873, 0.99319048, 0.99615873,
                                       0.9978254, 0.99865079, 0.99914286, 0.99946032, 0.99965079, 0.99969841,
                                       0.99971429, 0.99977778])
    align_latentSpace_thresh0 = np.array([0., 2.63157895, 5.26315789, 7.89473684, 10.52631579, 13.15789474,
                                          15.78947368, 18.42105263, 21.05263158, 23.68421053, 26.31578947, 28.94736842,
                                          31.57894737, 34.21052632, 36.84210526, 39.47368421, 42.10526316, 44.73684211,
                                          47.36842105, 50.])

    align_latentSpace_thresh1, align_latentSpace_acc1 = align_latentSpace_thresh0[8:], align_latentSpace_acc0[8:]
    Latent25D_ECCV2018 = np.array(
        [np.linspace(20, 50, 10), [0.9625, 0.9818, 0.9908, 0.9949, 0.9969, 0.9982, 0.9988, 0.9993, 0.9996, 0.9998]])
    spurra = np.array(
        [[21.0526, 23.6842, 26.3158, 28.9474, 31.5789, 34.2105, 36.8421, 39.4737, 42.1053, 44.7368, 47.3684, 50.0000],
         [0.9483, 0.9615, 0.9705, 0.9771, 0.9819, 0.9857, 0.9886, 0.9909, 0.9926, 0.9942, 0.9954, 0.9963]])

    gan = np.array([[19.1919, 22.2222, 25.2525, 28.2828, 31.3131, 34.3434, 37.3737, 40.4040, 43.4343, 46.4646, 49.4949],
                    [0.8713, 0.9035, 0.9271, 0.9446, 0.9574, 0.9670, 0.9741, 0.9795, 0.9833, 0.9867, 0.9895]])
    pso = np.array([[20, 25, 30, 35, 40, 45, 50],
                    [0.322368421052632, 0.539473684210526, 0.674342105263158, 0.756578947368421, 0.809210526315789,
                     0.865131578947368, 0.894736842105263]])
    icppso = np.array([[20, 25, 30, 35, 40, 45, 50],
                       [0.519736842105263, 0.644736842105263, 0.717105263157895, 0.773026315789474, 0.809210526315789,
                        0.848684210526316, 0.868421052631579]])
    chpr = np.array([[20, 25, 30, 35, 40, 45, 50],
                     [0.565789473684211, 0.717105263157895, 0.822368421052632, 0.881578947368421, 0.914473684210526,
                      0.9375, 0.960526315789474]])

    zb = np.array([[21.0526315789474, 23.6842105263158, 26.3157894736842, 28.9473684210526, 31.5789473684211,
                    34.2105263157895, 36.8421052631579, 39.4736842105263, 42.1052631578947, 44.7368421052632,
                    47.3684210526316, 50],
                   [0.869888888888889, 0.896873015873016, 0.916849206349206, 0.932142857142857, 0.943507936507937,
                    0.952753968253968, 0.959904761904762, 0.966047619047619, 0.971595238095238, 0.976547619047619,
                    0.980174603174603, 0.983277777777778]])

    forth = np.array([[22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 48, 50],
                      [0.612, 0.796, 0.8706666666666667, 0.892, 0.9226666666666666, 0.9493333333333334,
                       0.9593333333333334, 0.9793333333333333, 0.9953333333333333, 1.0, 1.0, 1.0, 1.0, 1.0]])
    auc_linlinYang = 0.996
    pso_auc = 'PSO (AUC=0.709)'
    icppso_auc = 'ICPPSO (AUC=0.748)'
    chpr_auc = 'CHPR (AUC=0.839)'
    forth_auc = 'Panteleris et al. (AUC=0.941)'
    zb_auc = 'Z&B (AUC=0.948)'
    gan_auc = 'Mueller et al. (AUC=0.965)'
    spurr_auc = 'Spurr et al. (AUC=0.983)'
    latent25D_auc = 'Iqbal (AUC=0.994)'
    linlinYang_leg = f'Yang et al. (AUC={auc_linlinYang:.3f})'
    color_list = get_color_list()
    pso_, = plt.plot(pso[0], pso[1], '-d', color=color_list[0])
    icppso_, = plt.plot(icppso[0], icppso[1], '-s', color=color_list[1])
    chpr_, = plt.plot(chpr[0], chpr[1], '-s', color=color_list[2])
    forth_, = plt.plot(forth[0], forth[1], '-o', color=color_list[3])
    zb_, = plt.plot(zb[0], zb[1], '-d', color=color_list[4])
    gan_, = plt.plot(gan[0], gan[1], '-s', color=color_list[5])
    spurra_, = plt.plot(spurra[0], spurra[1], '-s', color=color_list[6])
    latent25D_, = plt.plot(Latent25D_ECCV2018[0], Latent25D_ECCV2018[1], '-*', color=color_list[7])
    linlinYang_, = plt.plot(align_latentSpace_thresh1, align_latentSpace_acc1, '-o', color=color_list[8])
    ours2_, = plt.plot(ours_thresh_, ours_acc, '-s', color='r')
    ours2_leg = f'Ours (overall, G=3) (AUC={ours_auc:.3f})'


    plt.legend(handles=[pso_, icppso_, chpr_, forth_, zb_, gan_, spurra_, latent25D_, linlinYang_, ours2_],
               labels=[pso_auc, icppso_auc, chpr_auc, forth_auc, zb_auc, gan_auc, spurr_auc, latent25D_auc,
                       linlinYang_leg, ours2_leg], loc='lower right')
    # plt.tick_params(labelsize=15)
    plt.xlim(20, 50, 5)
    plt.ylim(0.3, 1, 0.1)
    plt.xlabel('Error Thresholds (mm)')
    plt.ylabel('3D PCK')
    plt.grid(True)
    plt.savefig(fname=save_dir, format="pdf")
    plt.show()


def plt_DO_3d(save_dir):
    ours_acc= np.array([0.0,0.12346086954882433,0.42027921424585835,0.6770373752284795,0.8383928242713419,
                        0.9270393119224651,0.9641591488650352,0.9793603687696114,0.988127485181775,
                        0.9933876358500513,0.9953495685402943,0.9964738191840027,0.997418682253104,
                        0.9978272184012672,0.9979579910446477,0.9981992455681699,0.9985249510885771,
                        0.9987188961574175,0.9988480534063682,0.9989126320308435])
    ours_auc =  0.862632577660143
    ### muller -- gan SOTA Res######################################
    thresh_others = np.array(
        [0, 5.0505, 10.1010, 15.1515, 20.2020, 25.2525, 30.3030, 35.3535, 40.4040, 45.4545, 50.5051, 55.5556,
         60.6061, 65.6566, 70.7071, 75.7576, 80.8081, 85.8586, 90.9091, 95.9596])
    sridhar_acc = [0, 0.0604, 0.2674, 0.4908, 0.6830, 0.7981, 0.8676, 0.9082, 0.9349, 0.9530, 0.9650, 0.9718, 0.9775,
                   0.9820,
                   0.9869, 0.9910, 0.9949, 0.9967, 0.9980, 0.9983]  ## RGBD
    AUC_sridhar = 0.8143
    gan_acc = np.array(
        [0, 0.0148, 0.0764, 0.1669, 0.2611, 0.3507, 0.4325, 0.5026, 0.5641, 0.6089, 0.6523, 0.6909, 0.7249, 0.7539,
         0.7782,
         0.8018, 0.8213, 0.8411, 0.8601, 0.8776])
    AUC_gan = 0.5579  ## RGB only

    thresh_latent25D = np.linspace(0, 100, 20)
    latent25D_wo_real2D_acc = np.array(
        [0.0000, 0.0171, 0.0708, 0.1443, 0.2287, 0.3116, 0.3901, 0.4643, 0.5327, 0.6025, 0.6689, 0.7262, 0.7768, 0.8201,
         0.8537, 0.8821, 0.9051, 0.9248, 0.9397, 0.9495])
    latent25D_all_acc = np.array(
        [0.0000, 0.0619, 0.1872, 0.3071, 0.4142, 0.5102, 0.6024, 0.6843, 0.7634, 0.8268, 0.8784, 0.9179, 0.9466, 0.9652,
         0.9775, 0.9845, 0.9881, 0.9925, 0.9950, 0.9964])
    AUC_latent25D = [0.565, 0.711]
    linlinyang_thresh = np.array([10.526315789473683, 15.789473684210527, 21.052631578947366,
                                  26.31578947368421, 31.578947368421055, 36.84210526315789, 42.10526315789473,
                                  47.36842105263158,
                                  52.63157894736842, 57.89473684210526, 63.15789473684211, 68.42105263157895,
                                  73.68421052631578,
                                  78.94736842105263, 84.21052631578947, 89.47368421052632, 94.73684210526316, 100.0])
    linlinyang_acc = np.array([0.245942, 0.362074, 0.475296, 0.565906, 0.64722, 0.726342, 0.791428, 0.840838, 0.8791236,
                               0.9097522, 0.9327236, 0.949314, 0.9659044, 0.9722854, 0.9812186, 0.9863234, 0.9914282,
                               0.993982])
    color_list = get_color_list()
    sridhar_, = plt.plot(thresh_others, sridhar_acc, '-o', color=color_list[0])
    gan_, = plt.plot(thresh_others, gan_acc, '-d', color=color_list[1])
    latent25D_wo_real2D_, = plt.plot(thresh_latent25D, latent25D_wo_real2D_acc, '-s', color=color_list[2])
    latent25D_all_, = plt.plot(thresh_latent25D, latent25D_all_acc, '-s', color=color_list[3])
    # print(thresh_latent25D.shape, linlinyang_acc.shape)
    linlinyang_, = plt.plot(linlinyang_thresh, linlinyang_acc, '-*', color=color_list[4])
    linlinyang_auc = 0.728
    sridhar_leg = f'Sridhar et al. (RGB-D) (AUC={AUC_sridhar:.2f})'
    gan_leg = f'Mueller et al. (RGB) (AUC={AUC_gan:.2f})'
    latent25D_wo_real2D_leg = f'Iqbal (RGB) w/o real-2D (AUC={AUC_latent25D[0]:.2f})'
    latent25D_all_leg = f'Iqbal (RGB) (AUC={AUC_latent25D[1]:.2f})'
    linlinyang_leg = f'Yang et al (AUC={linlinyang_auc:.2f})'
    ours_overallG3_, = plt.plot(thresh_latent25D, ours_acc, '-s', color='r')
    ours_overallG3_leg = f'Ours (overall, G=3) (AUC={ours_auc:.2f})'

    plt.xlim(0, 100, 20)
    plt.ylim(0, 1, 0.2)
    plt.grid(True)
    plt.legend(handles=[sridhar_, gan_, latent25D_wo_real2D_, latent25D_all_, linlinyang_, ours_overallG3_],
               labels=[sridhar_leg, gan_leg, latent25D_wo_real2D_leg, latent25D_all_leg, linlinyang_leg, ours_overallG3_leg], loc='best')
    plt.xlabel('Error Thresholds (mm)')
    plt.ylabel('3D PCK')
    plt.savefig(fname=save_dir, format="pdf")
    plt.show()

def plt_RHD_3D(save_dir):
    thresh_ = np.array(
        [21.05263158, 23.68421053, 26.31578947, 28.94736842, 31.57894737, 34.21052632, 36.84210526, 39.47368421,
         42.10526316, 44.73684211, 47.36842105, 50.])
    #  Ours Performance

    ours_acc = np.array([0.8815109621561235, 0.9092305543918447, 0.9294442117022763, 0.9446131825164081, 0.9554356933389192,
                         0.964407903924033, 0.9710410557184752, 0.9763126658287947, 0.9801180002792907, 0.9830680072615559, 0.9854594330400782, 0.987658846529814])
    ours_auc = 0.959544915006792

    ours_leg = f'Ours (overall, G=3) (AUC={ours_auc:.3f})'

    # Public avaliable data
    align_latentSpace_acc0 = np.array(
        [0., 0.09092813, 0.2198782, 0.36093807, 0.49493099, 0.61253904, 0.70552618, 0.77581183, 0.82908444, 0.8704741,
         0.89921304, 0.92200178,
         0.93959064, 0.95375944, 0.96321695, 0.9715577, 0.97776963, 0.98258563, 0.9857614, 0.98855328])
    align_latentSpace_thresh0 = np.array(
        [0., 2.63157895, 5.26315789, 7.89473684, 10.52631579, 13.15789474, 15.78947368, 18.42105263, 21.05263158,
         23.68421053, 26.31578947,
         28.94736842, 31.57894737, 34.21052632, 36.84210526, 39.47368421, 42.10526316, 44.73684211, 47.36842105, 50.])
    align_latentSpace_thresh1, align_latentSpace_acc1 = align_latentSpace_thresh0[8:], align_latentSpace_acc0[8:]
    auc_linlinYang = 0.943
    thresh_bf = np.array([21.05263158, 25, 30, 35, 40, 45, 50])
    zimmer_acc = np.array([0.44, 0.52697549, 0.62128873, 0.696, 0.74721685, 0.796, 0.82290621])
    cai_acc = np.array([0.73223468, 0.80055085, 0.8603462, 0.91354932, 0.93630473, 0.95564664, 0.96061099])
    spurr_acc = np.array([0.65347344, 0.73754589, 0.81706634, 0.87167887, 0.90558497, 0.93289152, 0.9411332])
    yang_acc = np.array([0.64323323, 0.73071947, 0.81706634, 0.87167887, 0.90558497, 0.93659812, 0.94738889])
    zimmer_auc, spurr_auc, Yang_auc, cai_auc = 0.675, 0.849, 0.849, 0.887
    color_list = get_color_list()
    zimmer_, = plt.plot(thresh_bf, zimmer_acc, '-d', color=color_list[0])
    spurr_, = plt.plot(thresh_bf, spurr_acc, '-s', color=color_list[1])
    Yang_, = plt.plot(thresh_bf, yang_acc, '-*', color=color_list[2])
    cai_, = plt.plot(thresh_bf, cai_acc, '-o', color=color_list[3])

    linlinYang_, = plt.plot(thresh_, align_latentSpace_acc1, '-o', color=color_list[4])

    zimmer_leg = f'Z&B (AUC={zimmer_auc:.3f})'
    spurr_leg = f'Spurr (AUC={spurr_auc:.3f})'
    Yang_leg = f'Yang (AUC={Yang_auc:.3f})'
    cai_leg = f'Cai (AUC={cai_auc:.3f})'

    linlinYang_leg = f'Yang et al. (AUC={auc_linlinYang:.3f})'
    ours2_, = plt.plot(thresh_, ours_acc, '-s', color='r')
    plt.xlim(19.5, 52, 5)
    plt.ylim(0.41, 1, 0.1)
    plt.legend(handles=[zimmer_, spurr_, Yang_, cai_, linlinYang_, ours2_],
               labels=[zimmer_leg, spurr_leg, Yang_leg, cai_leg, linlinYang_leg, ours_leg],
               loc='best')
    plt.xlabel('Error Thresholds (mm)')
    plt.ylabel('3D PCK')
    plt.grid(True)
    plt.savefig(fname=save_dir, format="pdf")
    plt.show()

if __name__ == '__main__':
    plt_STB('')
