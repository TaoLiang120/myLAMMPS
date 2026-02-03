import os
import numpy as np
import copy

from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpData

thisdata = lmpData.from_POSCAR("MgCu2.POSCAR", "atomic")
thisdata = thisdata.make_supercell([1, 1, 2])
thisdata.select_by_coords(xlim=[-0.01, 1], ylim=[-0.01, 1], zlim=[0.249, 0.75],
                         Fractional=True, style="INCLUDE", delete=True)
thisdata.modify_atoms(translation=[0, 0, -3], is_cartesian=True)
newmatrix = copy.deepcopy(thisdata.box.matrix)
newmatrix[2][2] /= 2
thisdata.modify_lmpbox(newmatrix, style=1)
thisdata.to_POSCAR("FeFe2.POSCAR")

