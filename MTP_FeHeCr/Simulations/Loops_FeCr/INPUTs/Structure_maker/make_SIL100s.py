import os,sys
import shutil
import copy
import numpy as np

from mylammps.inputs.data import lmpBox, lmpData

atom_style = "atomic"
force_field = ["Fe"]

a = 2.85
lvac = 7.0*a
thres = [0.03*a, 0.03*a, 0.03*a]

burgerm = a
bondlength = a*np.sqrt(3)/2

fname = "bcc_10x10x10.data"
basedata = lmpData.from_file(fname, atom_style)
basedata.scale_data(a, style=0)

i = 3
j = 2
k = 6
supercell = [i, j, k]
thisdata=basedata.make_supercell_simple(supercell)
thisdata.to_file("Sub010.dat")


fname = "bcc.POSCAR"
basedata = lmpData.from_POSCAR(fname, atom_style)
basedata.scale_data(a, style=0)
supercell = [i*10, 1, k*10]
thisdata=basedata.make_supercell(supercell)
thisdata.to_file("layer010.dat")

thisdata = lmpData.from_file("Sub010.dat", atom_style)
layerdata = lmpData.from_file("layer010.dat", atom_style)

rcut = a*0.5
check_distance=False
add_vacuum = False
direction = 2
typesoffset = False
centers=[[0.5, 0.5, 0.25],[0.5, 0.5, 0.5]]
rs = [1.7*a, 3.6*a, 7*a]
if typesoffset:
    ff_elements = ["Fe", "Fe"]
    atomic_masses=[55.845, 55.845]
else:
    ff_elements = ["Fe"]
    atomic_masses=[55.845]
for i in range(len(rs)):
    r = rs[i]
    outdata = thisdata.deepcopy()
    for icen in range(len(centers)):
        center = centers[icen]

        inds, thisxyzs, ds, itypes = outdata.select_by_radius(bondlength+0.03*a, center=center,
                                                      is_cartesian=False, depress=None,
                                                      delete=False, style=1, sort=True)
        subcen = thisxyzs[0]

        tmpdata = layerdata.deepcopy()
        tmpdata.select_by_radius(r, center=[0.5, 0.5, 0.5], is_cartesian=False, depress=1, delete=True, style=0)

        xyzs = np.vstack((tmpdata.atoms["x"], tmpdata.atoms["y"], tmpdata.atoms["z"]))
        coords = [np.mean(xyzs[0]), np.mean(xyzs[1]), np.mean(xyzs[2])]
        inds, thisxyzs, ds, itypes = tmpdata.select_by_radius(bondlength+0.03*a, center=coords,
                                                             is_cartesian=True, depress=None,
                                                             delete=False, style=1, sort=True)
        layercen = thisxyzs[0] 

        translation = subcen - layercen
        translation[0] -= a/4
        translation[1] -= a/2
        translation[2] -= 0.0

        outdata = outdata.mergy_data(tmpdata, translation=translation, rotation=None, depress=None,
                      is_cartesian=True, normalization4symmop=False,
                      normalization=True, check_distance=check_distance, rcut=rcut,
                      modify_box=False, newmatrix=None, style=0,
                      add_vacuum=add_vacuum, lvac=lvac, direction=direction,
                      typesoffset=typesoffset, ff_elements=ff_elements, atomic_masses=atomic_masses,
                      reset_ids=True)

        print(subcen)
        print(layercen)
        print(translation)
        print(f"finished icen:{icen} ")
        print("=====")
    print(f"finished radius:{r} ")
    print("=====================")

    fout = "SIL010_r" + str(r) + ".dat"
    outdata.to_file(fout)
    


