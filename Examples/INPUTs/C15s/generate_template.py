import os
import numpy as np
import copy
from pymatgen.core.operations import SymmOp
from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpData
from mylammps.inputs.util import mat_lengths, mat_angles, generate_rotation_matrix

def generate_templates(thisdata):
    cart_coords = np.vstack((thisdata.atoms['x'], thisdata.atoms['y'], thisdata.atoms['z']))
    cart_coords = - cart_coords
    cart_coords += 6.0
    thisdata.atoms['x'] = cart_coords[0]
    thisdata.atoms['y'] = cart_coords[1]
    thisdata.atoms['z'] = cart_coords[2]
    thisdata.swap_axes([0, 1, 2])
    thisdata.coords2fracts()
    return thisdata


fnames = ["Z16_0.POSCAR", "Z12_0.POSCAR"]
outfnames = ["Z16_1.POSCAR", "Z12_1.POSCAR"]
for ifile in range(len(fnames)):
    fname = fnames[ifile]
    fout = outfnames[ifile]
    thisdata = lmpData.from_POSCAR(fname, "atomic")
    thisdata = generate_templates(thisdata)
    thisdata.to_POSCAR(fout)


