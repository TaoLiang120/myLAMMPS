import numpy as np
import copy
import itertools
from numpy import pi

from mylammps.inputs.data import lmpData, lmpBox
atom_style = "atomic"


fname = "bcc_10x10x10.data"
basedata = lmpData.from_file(fname, atom_style)
a = 1.0
basedata.scale_data(a, style=0)
bondlength = a * np.sqrt(3) / 2
radius = a + 0.02
supercell = [2, 2, 2]
thisdata = basedata.make_supercell_simple(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5-0.5/10/2, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
isorts = np.argsort(ds)
inds = inds[isorts]
xyzs = xyzs[isorts]
cs0 = xyzs[0]
todel = [inds[0]]
for i in range(1, len(inds)):
    cs = xyzs[i]-cs0
    d = np.linalg.norm(cs)
    if d > bondlength - 0.01 and d<bondlength+0.01:
        if xyzs[i][1] > cs0[1]:
            todel.append(inds[i])
            print(i, xyzs[i], cs[0])
            break
print(todel)
thisdata.remove_by_inds(todel)
thisdata.reset_atom_ids()
thisdata.to_file("divacancy_1nn.dat")
print("finished divacancy_1nn")


fname = "bcc_10x10x10.data"
basedata = lmpData.from_file(fname, atom_style)
a = 1.0
basedata.scale_data(a, style=0)
bondlength = a * np.sqrt(3) / 2
radius = a + 0.02
supercell = [2, 2, 2]
thisdata = basedata.make_supercell_simple(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5/10/2, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
isorts = np.argsort(ds)
inds = inds[isorts]
xyzs = xyzs[isorts]
cs0 = xyzs[0]
todel = [inds[0]]
for i in range(1, len(inds)):
    cs = xyzs[i]-cs0
    d = np.linalg.norm(cs)
    if d > a - 0.01 and d<a+0.01:
        if xyzs[i][1] > cs0[1]:
            todel.append(inds[i])
            print(i, xyzs[i], cs[0])
            break
print(todel)
thisdata.remove_by_inds(todel)
thisdata.reset_atom_ids()
thisdata.to_file("divacancy_2nn.dat")
print("finished divacancy_2nn")


fname = "bcc_10x10x10.data"
basedata = lmpData.from_file(fname, atom_style)
a = 1.0
basedata.scale_data(a, style=0)
bondlength = a * np.sqrt(3) / 2
radius = a + 0.02
supercell = [2, 2, 2]
thisdata = basedata.make_supercell_simple(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
isorts = np.argsort(ds)
inds = inds[isorts]
xyzs = xyzs[isorts]
cs0 = xyzs[0]
todel = [inds[0]]
for i in range(1, len(inds)):
    cs = xyzs[i]-cs0
    d = np.linalg.norm(cs)
    if d< bondlength+0.01:
        todel.append(inds[i])
        print(i, xyzs[i], cs[0])
print(todel)
thisdata.remove_by_inds(todel)
thisdata.reset_atom_ids()
thisdata.to_file("9vac.dat")
print("finished 9vac")
