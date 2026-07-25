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
supercell = [[1, 1, -2], [1, 1, 1], [1, -1, 0]]

outdata=thisdata.make_supercell(supercell)
newmatrix = copy.deepcopy(outdata.box.matrix)
newmatrix[1][1] /= 2.0
outdata.modify_lmpbox(newmatrix, style=2)
outdata.to_file("bcc112_111_110.data")

fname = "bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
supercell = [[1, 1, -2], [1, 1, 1], [1, -1, 0]]

outdata=thisdata.make_supercell(supercell)
outdata.to_file("bcc112_111_110_full.data")

fname = "octa_bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
supercell = [[1, 1, -2], [1, 1, 1], [1, -1, 0]]

outdata=thisdata.make_supercell(supercell)
newmatrix = copy.deepcopy(outdata.box.matrix)
newmatrix[1][1] /= 2.0
outdata.modify_lmpbox(newmatrix, style=2)
outdata.to_file("octa_112_111_110.data")

fname = "bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
supercell = [10, 10, 10]
outdata=thisdata.make_supercell(supercell)
outdata.to_file("bcc_10x10x10.data")


fname = "bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
outdata.to_file("bcc.data")

fname = "tetra_bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
outdata.to_file("tetra_bcc.data")

fname = "octa_bcc.POSCAR"
thisdata = lmpData.from_POSCAR(fname, atom_style, ff_elements=ff_elements, is_sort=is_sort)
outdata.to_file("octa_bcc.data")