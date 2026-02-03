import os, sys
import numpy as np
import copy
from numpy import pi

from mylammps.inputs.data import lmpData

atom_style = "atomic"
ff_elements = ["Fe"]
is_sort = False

fname = "bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
supercell = [[1, -1, 0], [1, 1, -2], [1, 1, 1]]

outdata=thisdata.make_supercell(supercell)
newmatrix = copy.deepcopy(outdata.box.matrix)
newmatrix[2][2] /= 2.0
outdata.modify_lmpbox(newmatrix, style=2)
outdata.reset_atom_ids()
outdata.to_file("bcc110_112_111.data")

