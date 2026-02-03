import numpy as np
import copy
from scipy.optimize import leastsq

from ase.eos import EquationOfState
from pymatgen.core.lattice import Lattice
from mylammps.myglobal import Constants

eVA2GPa = Constants["eVA2GPa"]


def _general_function(params, xdata, ydata, function):
    return function(xdata, *params) - ydata


def curve_fit(f, x, y, p0):
    func = _general_function
    args = (x, y, f)
    popt, pcov, infodict, mesg, ier = leastsq(
        func, p0, args=args, full_output=True, maxfev=200000, ftol=1.0e-15, xtol=1.0e-15)

    if ier not in [1, 2, 3, 4]:
        raise RuntimeError("Optimal parameters not found: " + mesg)
    # end of this part
    return popt, pcov, infodict, mesg, ier


def ca_fit(x, y, n):
    # Basic polynomial fit whose accuracy assumes that there is no bad points:
    z = np.polyfit(x, y, n)
    fit = np.poly1d(z)
    xmin = -z[1] / (2.0 * z[0])
    ymin = fit(xmin)
    return xmin, ymin


def distortion_poly1(x, a2):
    E = a2 * x ** 2
    return E


def distortion_poly2(x, a2, a0):
    E = a2 * x ** 2 + a0
    return E


def distortion_poly3(x, a2, a1, a0):
    E = a2 * x ** 2 + a1 * x + a0
    return E


def distortion_fit(x, y, num=2):
    if num == 1:
        ydata = y - y[0]
    else:
        ydata = y

    # Starting estimates for the actual fit
    z = np.polyfit(x, ydata, 2)

    if num == 1:
        p0 = [z[0]]
    elif num == 2:
        p0 = [z[0], z[2]]
    else:
        p0 = [z[0], z[1], z[2]]

    popt, pcov, infodict, mesg, ier = curve_fit(eval('distortion_poly{0}'.format(num)), x, ydata, p0)

    # Compute error estimate for the fit
    residuals = infodict['fvec']
    chisqr = np.sum(residuals ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    rsquared = 1.0 - chisqr / ss_tot

    # Make sure the return variable is always an array
    if isinstance(popt, type(np.sqrt(1.0))):
        popt = np.array([popt])
    return popt, rsquared


class Elastic_properties:
    @staticmethod
    def EOS_fit(volumes, energies, eos="sj"):
        EOS = EquationOfState(volumes, energies, eos=eos)
        v0, e0, B = EOS.fit()
        B *= eVA2GPa
        redict = {"B": B, "v0": v0, "e0": e0}
        return redict

    @staticmethod
    def EOS_fit4hcp(boxss, energiess, eos="sj"):
        """
        :param boxss: first index on a, second index on coa (c over a). The relaxed must locate at [0,0]
        :param energiess: first index is changing on a, second index is change on coa
        :param eos: eos style
        :return: B, v0, e0, B, a0, ca0, r, cs0
        """

        boxss = np.array(boxss)
        energiess = np.array(energiess)
        thisshape = energiess.shape

        caorder = 2
        ca_list = []
        a_list = []
        v_list = []
        volumes = np.zeros((thisshape[0], thisshape[1]))
        #energies = np.zeros((thisshape[0], thisshape[1]))
        #energies0 = []
        cas0 = []

        for i in range(thisshape[0]):
            caengs = []
            thiscas = []
            for jca in range(thisshape[1]):
                thislatt = Lattice(boxss[i, jca, :, :])
                thisvol = thislatt.volume
                thisen = energiess[i, jca]
                if jca == 0:
                    a_list.append(thislatt.lengths[0])
                    v_list.append(thisvol)
                    #energies0.append(thisen)
                thiscas.append(thislatt.lengths[2] / thislatt.lengths[0])
                caengs.append(thisen)
                volumes[i, jca] = thisvol
                #energies[i, jca] = thisen

            ca0, en0 = ca_fit(thiscas, caengs, caorder)
            if i == 0: ca_list = copy.deepcopy(thiscas)
            cas0.append(ca0)

        EOS = EquationOfState(volumes.flatten(), energiess.flatten(), eos=eos)
        v0, e0, B = EOS.fit()
        B *= eVA2GPa
        ca0_vs_v = np.polyfit(v_list, cas0, 2)
        ca0_vs_v = np.poly1d(ca0_vs_v)
        c_over_a0 = ca0_vs_v(v0)
        dca0_vs_dv = np.polyder(ca0_vs_v, 2)
        dca0dv0 = dca0_vs_dv(v0)
        r0 = -v0 / c_over_a0 * dca0dv0

        energies_cs = []
        for i in range(thisshape[1]):
            csengs = []
            csvols = []
            for j in range(thisshape[0]):
                csvols.append(volumes[j][i])
                csengs.append(energiess[j][i])
            eos = EquationOfState(csvols, csengs)
            v_tmp, e_tmp, B_tmp = eos.fit()
            energies_cs.append(e_tmp)

        e_vs_ca = np.polyfit(ca_list, energies_cs, 2)
        e_vs_ca = np.poly1d(e_vs_ca)
        d2e_vs_dca2 = np.polyder(e_vs_ca, 2)
        d2edca02 = d2e_vs_dca2(c_over_a0)
        cs0 = 9.0 * c_over_a0 ** 2 / 2.0 / v0 * d2edca02 * eVA2GPa
        a0 = 2.0 * v0 / np.sqrt(3) / c_over_a0
        a0 = np.power(a0, 1.0 / 3.0)
        redict = {"B": B, "v0": v0, "e0": e0, "a0": a0, "coa0": c_over_a0, "r0": r0, "cs0": cs0}
        return redict

    @staticmethod
    def distort_fit(deltas, energies):
        deltas = np.array(deltas)
        energies = np.array(energies)
        c_fit, c_rsq = distortion_fit(deltas, energies)
        redict = {"c_fit": c_fit, "c_rsq": c_rsq}
        return redict

    @staticmethod
    def compute_cubic_elastic_constants(B, c11, c12, c44):
        # Voigt average
        BV = (c11 + 2 * c12) / 3.0
        GV = (c11 - c12 + 3 * c44) / 5.0
        EV = 9 * BV * GV / (3 * BV + GV)
        vV = (3 * BV - 2 * GV) / (6 * BV + 2 * GV)

        # Reuss average
        BR = BV
        GR = 5 * (c11 - c12) * c44 / (4 * c44 + 3 * (c11 - c12))
        ER = 9 * BR * GR / (3 * BR + GR)
        vR = (3 * BR - 2 * GR) / (6 * BR + 2 * GR)

        # Hill average
        BH = (BV + BR) / 2.0
        GH = (GV + GR) / 2.0
        EH = 9 * BH * GH / (3 * BH + GH)
        vH = (3 * BH - 2 * GH) / (6 * BH + 2 * GH)
        # Elastic anisotropy
        # AVR = (GV - GR) / (GV + GR)
        AU = 5.0 * GV / GR + BV / BR - 6.0

        G = (GV + GR) / 2
        Y = 9.0 * B * G / (3 * B + G)
        AZ = 2 * G / (c11 - c12)
        v = 0.5 * (3 * B - 2.0 * G) / (3 * B + G)
        print(f"c11:{c11} c12:{c12} c44:{c44}")
        print(f"Bulk:{B} Shear:{G} Youngs:{Y} Poisson:{v}")
        print(f"voigt Shear:{GV} Youngs:{EV} Poisson:{vV}")
        print(f"Reuss Shear:{GR} Youngs:{ER} Poisson:{vR}")
        print(f"Hill Shear:{GH} Youngs:{EH} Poisson:{vH}")
        print(f"Au:{AU} Az:{AZ}")
        print(f"=====")
        redict = {"B": B, "c11": c11, "c12": c12, "c44": c44, "G": G, "Y": Y, "v": v, "AU": AU, "AZ": AZ}
        return redict

    @staticmethod
    def get_cubic_elastic_constants(B, V0, pOD, pMD):
        cprime = pOD[0] / 2.0 / V0 * eVA2GPa
        c44 = pMD[0] / 2.0 / V0 * eVA2GPa
        c11 = B + 4.0 / 3.0 * cprime
        c12 = B - 2.0 / 3.0 * cprime
        redict = {"B": B, "v0": V0}
        redict = Elastic_properties.compute_cubic_elastic_constants(B, c11, c12, c44)
        redict["v0"] = V0
        return redict

    @staticmethod
    def compute_hcp_elastic_constants(B, cs0, c11, c33, c12, c13, c44, c66, c2):
        BV = (2 * c11 + 2 * c12 + 4 * c13 + c33) / 9.0
        GV = (12 * c44 + 12 * c66 + cs0) / 30.0
        EV = 9 * BV * GV / (3 * BV + GV)
        vV = (3 * BV - 2 * GV) / (6 * BV + 2 * GV)

        # Reuss average
        BR = B
        GR = 5.0 / 2.0 * (c44 * c66 * c2) / ((c44 + c66) * c2 + 3.0 * BV * c44 * c66)
        ER = 9 * BR * GR / (3 * BR + GR)
        vR = (3 * BR - 2 * GR) / (6 * BR + 2 * GR)

        # Hill average
        BH = (BV + BR) / 2.0
        GH = (GV + GR) / 2.0
        EH = 9 * BH * GH / (3 * BH + GH)
        vH = (3 * BH - 2 * GH) / (6 * BH + 2 * GH)

        # Elastic anisotropy
        # AVR = (GV - GR) / (GV + GR)
        AU = 5 * GV / GR + BV / BR - 6.0
        G = (GV + GR) / 2.0
        Y = 9.0 * B * G / (3 * B + G)
        v = 0.5 * (3 * B - 2.0 * G) / (3 * B + G)
        AZ = 2 * G / (c11 - c12)

        print(f"c11:{c11} c12:{c12} c44:{c44} c13:{c13} c33:{c33}")
        print(f"Bulk:{B} Shear:{G} Youngs:{Y} Poisson:{v}")
        print(f"---")
        print(f"voigt Bulk:{BV} Shear:{GV} Youngs:{EV} Poisson:{vV}")
        print(f"Reuss Bulk:{BR} Shear:{GR} Youngs:{ER} Poisson:{vR}")
        print(f"Hill Bulk:{BH} Shear:{GH} Youngs:{EH} Poisson:{vH}")
        print(f"Au:{AU} Az:{AZ}")
        print(f"=====")
        redict = {"B": B, "c11": c11, "c12": c12, "c44": c44, "c13": c13, "c33": c33, "G": G, "Y": Y, "v": v,
                  "AU": AU, "AZ": AZ}
        return redict

    @staticmethod
    def get_hcp_elastic_constance(B, V0, r0, cs0, pOD, pMD, pTD=None):
        if pTD is not None:
            cs0 = pTD[0] / V0 * eVA2GPa
        c66 = pOD[0] / 2.0 / V0 * eVA2GPa
        c44 = pMD[0] / 2.0 / V0 * eVA2GPa

        c11 = B + c66 + cs0 * (2 * r0 - 1) ** 2 / 18.0
        c12 = B - c66 + cs0 * (2 * r0 - 1) ** 2 / 18.0
        c13 = B + 1.0 / 9.0 * cs0 * (2 * r0 ** 2 + r0 - 1)
        c33 = B + 2.0 / 9.0 * cs0 * (r0 + 1) ** 2
        c2 = c33 * (c11 + c12) - 2.0 * c13 ** 2

        redict = Elastic_properties.compute_hcp_elastic_constants(B, cs0, c11, c33, c12, c13, c44, c66, c2)
        redict["v0"] = V0
        return redict
