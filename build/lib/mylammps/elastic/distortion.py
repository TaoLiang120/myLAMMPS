import numpy as np


class Distortion:
    @staticmethod
    def apply_distortion(dist_mat, mat):
        array = np.array(mat)
        for i in range(len(mat)):
            array[i, :] = np.array([np.sum(dist_mat[0, :] * mat[i, :]),
                                    np.sum(dist_mat[1, :] * mat[i, :]),
                                    np.sum(dist_mat[2, :] * mat[i, :])])
        return array

    @staticmethod
    def EOS_dis(ds):
        da = np.eye(3, dtype=float)
        da[0][0] += ds
        da[1][1] += ds
        da[2][2] += ds
        return da

    @staticmethod
    def EOS_dis_hcp(dsa, dsca, ca0):
        da = np.eye(3, dtype=float)
        da[0][0] += dsa
        da[1][0] = -0.5 * (1 + dsa)
        da[1][1] = 0.5 * np.sqrt(3.0) * (1 + dsa)
        da[2][2] = ca0 * (1 + dsca)
        return da

    @staticmethod
    def tetr_dis(ds):
        da = np.eye(3, dtype=float)
        da[0][0] += ds
        da[1][1] += ds
        da[2][2] = 1.0 / ((1.0 + ds) * (1.0 + ds))
        return da

    @staticmethod
    def orth_dis(ds):
        da = np.eye(3, dtype=float)
        da[0][0] += ds
        da[1][1] -= ds
        da[2][2] = 1.0 / (1.0 - ds * ds)
        return da

    @staticmethod
    def mono_dis(ds, crystal='bcc'):
        da = np.eye(3, dtype=float)
        if "HCP" in crystal.upper():
            da[0][2] = ds
            da[2][0] = ds
            da[2][2] = 1.0 / (1.0 - ds * ds)
        else:
            da[0][1] = ds
            da[1][0] = ds
            da[2][2] = 1.0 / (1.0 - ds * ds)
        return da
