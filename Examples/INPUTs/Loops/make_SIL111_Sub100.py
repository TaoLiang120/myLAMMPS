import os,sys
import shutil
import copy
import numpy as np
from numpy import pi

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices
from mylammps.inputs.util import generate_rotation_matrix

atom_style = "atomic"
force_field = ["Fe"]

a = 2.85
lvac = 7.0*a
thres = [0.03*a, 0.03*a, 0.03*a]

burgerm = a*np.sqrt(3)/2
bondlength = a*np.sqrt(3)/2

fname = "bcc_10x10x10.data"
basedata = lmpData.from_file(fname, atom_style)
basedata.scale_data(a, style=0)

i = 3
j = 2
k = 3
supercell = [i, j, k]
thisdata=basedata.make_supercell_simple(supercell)
thisdata.to_file("Sub010.dat")

fname = "bcc112_111_110.data"
basedata = lmpData.from_file(fname, atom_style)
basedata.scale_data(a, style=0)
i = 7
k = 11
supercell = [i, 1, k]
thisdata=basedata.make_supercell_simple(supercell)
thisdata.to_file("layer111.dat")

thisdata = lmpData.from_file("Sub010.dat", atom_style)
layerdata = lmpData.from_file("layer111.dat", atom_style)
subm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

rcut = bondlength+0.03*a
typesoffset = False
check_distance=True
add_vacuum = False
direction = 2
centers = [[0.5, 0.5, 0.5]] #, [0.5, 0.5, 0.65]]
layms = [np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]]),
         np.array([[2, 1, 1], [-1, 1, 1], [0, 1, -1]])]

if typesoffset:
    ff_elements = ["Fe", "Fe"]
    atomic_masses=[55.845, 55.845]
else:
    ff_elements = ["Fe"]
    atomic_masses=[55.845]

rs = [7*a]
for i in range(len(rs)):
    r = rs[i]
    outdata = thisdata.deepcopy()
    for icen in range(len(centers)):
        center = centers[icen]
        laym = layms[icen]
        symmop = find_symmop_lattices(Lattice(subm), Lattice(laym))

        tmpdata = layerdata.deepcopy()
        laycen = np.array([0.5, 0.5, 0.5])
        laycen = np.dot(laycen.T, tmpdata.box.matrix)
        laycen = laycen.T
        laycen[1] = bondlength/3
        ind, laycen, d, itypes = tmpdata.find_center_atom_coords(bondlength + 0.03*a,
                   center=laycen, is_cartesian=True, style=1)
        tmpdata.select_by_radius(r, center=laycen, is_cartesian=True, depress=1, delete=True, style=1)
        tmpdata.atoms = lmpData.modify_by_symmetry(tmpdata.atoms, symmop, normalization=True)

        outdata = outdata.merge_data_with_splits(tmpdata, bondlength,
                              to_center=center, is_cartesian=False,
                              splits=[0.0, -0.25, -0.25], rcut=0.3*a, tolerance=0.1*a,
                              modify_box=False, newmatrix=None, style=0,
                              add_vacuum=False, lvac=lvac, direction=2,
                              typesoffset=typesoffset, ff_elements=ff_elements, atomic_masses=atomic_masses,
                              reset_ids=True)

        print(f"finished icen:{icen} ")
        print(tmpdata.natoms)
        print("=====")
    print(f"finished radius:{r} ")
    print("======================")
    fout = "SIL111_Sub010_r" + str(r) + ".dat"
    outdata.to_file(fout)
    
    


