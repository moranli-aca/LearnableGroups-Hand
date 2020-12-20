import numpy as np
class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_pred):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        #keypoint_gt = np.squeeze(keypoint_gt)
        #keypoint_pred = np.squeeze(keypoint_pred)

        assert len(keypoint_gt.shape) == 3
        assert len(keypoint_pred.shape) == 3

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=2))

        num_kp = keypoint_gt.shape[1]
        for i in range(num_kp):
            self.data[i].append(euclidean_dist[:, i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id]) ### 1364,Bz
        # print('datatype', data.shape)
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds


class EvalUtil_SingleFrame:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=5):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_pred,keypoint_vis):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        keypoint_gt = np.squeeze(keypoint_gt)
        keypoint_pred = np.squeeze(keypoint_pred)
        #         keypoint_vis = np.squeeze(keypoint_vis).astype('bool')
        # print(keypoint_gt.shape, keypoint_pred.shape, keypoint_vis.shape)

        assert len(keypoint_gt.shape) == 2
        assert len(keypoint_pred.shape) == 2
        #         assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
        #         print(euclidean_dist.shape, 'diffshape')
        num_kp = keypoint_gt.shape[0]
        #         print(num_kp, 'knum')
        for i in range(num_kp):
            #             print(i, 'kpt_id')
            if keypoint_vis[i]==1:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds
def calc_auc(x, y):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    integral = np.trapz(y, x)
    norm = np.trapz(np.ones_like(y), x)

    return integral / norm


def get_stb_ref_curves():
    """
        Returns results of various baseline methods on the Stereo Tracking Benchmark Dataset reported by:
        Zhang et al., ‘3d Hand Pose Tracking and Estimation Using Stereo Matching’, 2016
    """
    curve_list = list()
    thresh_mm = np.array([20.0, 25, 30, 35, 40, 45, 50])
    pso_b1 = np.array([0.32236842,  0.53947368,  0.67434211,  0.75657895,  0.80921053, 0.86513158,  0.89473684])
    curve_list.append((thresh_mm, pso_b1, 'PSO (AUC=%.3f)' % calc_auc(thresh_mm, pso_b1)))
    icppso_b1 = np.array([ 0.51973684,  0.64473684,  0.71710526,  0.77302632,  0.80921053, 0.84868421,  0.86842105])
    curve_list.append((thresh_mm, icppso_b1, 'ICPPSO (AUC=%.3f)' % calc_auc(thresh_mm, icppso_b1)))
    chpr_b1 = np.array([ 0.56578947,  0.71710526,  0.82236842,  0.88157895,  0.91447368, 0.9375,  0.96052632])
    curve_list.append((thresh_mm, chpr_b1, 'CHPR (AUC=%.3f)' % calc_auc(thresh_mm, chpr_b1)))
    return curve_list

