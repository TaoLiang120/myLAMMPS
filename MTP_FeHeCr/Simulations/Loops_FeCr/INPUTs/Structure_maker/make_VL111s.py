import os,sys
import shutil
import copy
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices
from mylammps.inputs.util import generate_rotation_matrix

atom_style = "atomic"
force_field = ["Fe"]

a = 1.0 #2.84 #2.83037145
burgerm = a*np.sqrt(3)/2
bondlength = a*np.sqrt(3)/2

fname = "bcc112_111_110.data"
basedata = lmpData.from_file(fname, atom_style)
basedata.scale_data(a, style=0)

i = 20
j = 40
k = 30
supercell = [i, j, k]
thisdata=basedata.make_supercell(supercell)
thisdata.to_file("Sub111.dat")

fname = "bcc112_111_110.data"
basedata = lmpData.from_file(fname, atom_style)
basedata.scale_data(a, style=0)
supercell = [i, 1, k]
thisdata=basedata.make_supercell(supercell)
#newaxis = [1, 2, 0]
#thisdata.swap_axes(newaxis)
thisdata.to_file("layer111.dat")

thisdata = lmpData.from_file("Sub111.dat", atom_style)
layerdata = lmpData.from_file("layer111.dat", atom_style)

subm = np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]])
layms = [
         np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]),
         np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]]),
         np.array([[2, 1, 1], [-1, 1, 1], [0, 1, -1]])
        ]

centers = [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]]
splitss = [[-0.15, -0.1, -0.3], [0, -0.23, -0.23]]
'''
note that the second SIL has the identical orientation with substrate.
this second splits on the second SIL, visually has 111/2 but has higher energy
'''
#splitss = [[-0.15, -0.1, -0.3], [-0.15, -0.1, -0.3]]
'''
note that the second SIL has the identical orientation with substrate.
this second splits on the second SIL, visually has 100/2 but has lower energy
'''

subm = np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]])
layms = [
         np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]]),
        ]

centers = [[0.5, 0.5, 0.5]] #, [0.5, 0.5, 0.75]]
splitss = [[0.0, -0.25, -0.25]] #, [0, -0.23, -0.23]]

mergy_style=1
rcut = bondlength + 0.03*a
tolerance = bondlength - 0.03*a
add_vacuum = False
direction = 2
depresss = [1, 1] 
typesoffset = False
check_distance=True

rs = [6*a]
if typesoffset:
    ff_elements = ["Fe", "He", "Cr"]
    atomic_masses=[55.845, 4.0026, 51.9961]
else:
    ff_elements = ["Fe", "He", "Cr"]
    atomic_masses=[55.845]

for irad in range(len(rs)):
    r = rs[irad]
    outdata = thisdata.deepcopy()
    for icen in range(len(centers)):
        center = centers[icen]
        laym = layms[icen]
        symmop = find_symmop_lattices(Lattice(subm), Lattice(laym))
        tmpdata = layerdata.deepcopy()

        laycen = np.array([0.5, 0.5, 0.5])
        laycen = np.dot(laycen.T, tmpdata.box.matrix)
        laycen = laycen.T
        laycen[depresss[icen]] = bondlength/3
        ind, laycen, d, itypes = tmpdata.find_center_atom_coords(bondlength + 0.1, 
                                     center=laycen, is_cartesian=True, style=1)
        tmpdata.select_by_radius(r, center=laycen, is_cartesian=True, depress=depresss[icen], delete=True, style=1)
        tmpdata.atoms = lmpData.modify_by_symmetry(tmpdata.atoms, symmop, normalization=True)
        outdata = outdata.merge_data_with_splits(tmpdata, bondlength,
                                                 mergy_style=mergy_style,
                                  to_center=center,  is_cartesian=False,
                                  splits=splitss[icen], rcut=0.2*a, tolerance=0.1*a,
                                  modify_box=False, newmatrix=None, style=0,
                                  add_vacuum=False, lvac=20.0, direction=2,
                                  typesoffset=typesoffset, ff_elements=ff_elements, atomic_masses=atomic_masses,
                                  reset_ids=True)



        print(tmpdata.natoms)
        print(f"finished icen:{icen} ")
        print("=====")
    print(f"finished radius:{r} ")
    print("=====================")
    fout = "VL111_r" + str(r/a) + ".dat"
    outdata.to_file(fout)
    

