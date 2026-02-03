import os
import numpy as np
import copy
from pymatgen.core.operations import SymmOp
from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpData
from mylammps.inputs.util import mat_lengths, mat_angles, generate_rotation_matrix


fname = "bcc_2x2x2.POSCAR"
thisdata = lmpData.from_POSCAR(fname, "atomic")
thisdata.coords2fracts(normalization=True, style=0)
print(thisdata.atoms)
xyzns = np.vstack((thisdata.atoms['xsn'], thisdata.atoms['ysn'], thisdata.atoms['zsn']))
xyzns *= 4
ds = np.sum(xyzns*xyzns, axis=0)
inds = np.argsort(ds)
xyzns = xyzns[:, inds]
xyzns = xyzns.astype(int)
xyzns = xyzns.T
xyzns = xyzns.tolist()
print(xyzns)

m = [4, 5, -3, -4, -5]
for i in m:
    print(i, i%-4)
