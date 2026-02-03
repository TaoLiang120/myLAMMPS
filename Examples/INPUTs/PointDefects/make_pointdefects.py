import numpy as np
import copy
import itertools
from numpy import pi

from mylammps.inputs.data import lmpData, lmpBox

a = {"1": 0, "2": 1}
print(len(a))
print("000")
fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 3.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
supercell = [4, 4, 4]
thisdata = basedata.make_supercell(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)

cs = thisdata.atoms.iloc[inds[0]].to_dict()
thisdata.remove_by_inds(inds[0], style="itag")
indict = {"type": 1, "x": cs["x"], "y": cs["y"]+b/3, "z": cs["z"]+b/3}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
indict = {"type": 1, "x": cs["x"], "y": cs["y"]-b/3, "z": cs["z"]-b/3}
thisdata.add_an_entry(indict)
thisdata.reset_atom_ids()
thisdata.to_file("dumbbell011.dat")
print("finished dumbbell011")


'''
fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 3.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
supercell = [20, 20, 20]
thisdata = basedata.make_supercell(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
cs = thisdata.atoms.iloc[inds[0]].to_dict()
thisdata.remove_by_inds(inds[0], style="itag")
indict = {"type": 1, "x": cs["x"]+b/4, "y": cs["y"]+b/4, "z": cs["z"]+b/4}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
indict = {"type": 1, "x": cs["x"]-b/4, "y": cs["y"]-b/4, "z": cs["z"]-b/4}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
thisdata.reset_atom_ids()
thisdata.to_file("dumbbell111.dat")
print("finished dumbbell111")



fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 3.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
supercell = [20, 20, 20]
thisdata = basedata.make_supercell(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
cs = thisdata.atoms.iloc[inds[0]].to_dict()
thisdata.remove_by_inds(inds[0], style="itag")
indict = {"type": 1, "x": cs["x"]+b/2.5, "y": cs["y"], "z": cs["z"]}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
indict = {"type": 1, "x": cs["x"]-b/2.5, "y": cs["y"], "z": cs["z"]}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
thisdata.reset_atom_ids()
thisdata.to_file("dumbbell100.dat")
print("finished dumbbell100")

fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 3.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
n = 20
flength = 0.5/n
supercell = [n, n, n]
thisdata = basedata.make_supercell(supercell)
center = np.array([0.5, 0.5-flength, 0.5])
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=center,
                                                  is_cartesian=False, delete=False, sort=True)
for i in inds:
    thisdict = thisdata.atoms.iloc[i].to_dict()
    print(thisdict)
cs = np.dot(thisdata.box.matrix.T, center)

for i in inds:
    thisdict = thisdata.atoms.iloc[i].to_dict()
    thiscoords = np.array([thisdict['x'], thisdict['y'], thisdict['z']])
    diff = cs - thiscoords
    d = np.linalg.norm(diff)
    print(f"i: {i} distance:{d}")
    print("---")

indict = {"type": 1, "x": cs[0], "y": cs[1], "z": cs[2]}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
thisdata.reset_atom_ids()
thisdata.to_file("SIA_Oct.dat")
print("finished SIA_Oct")



fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 3.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
n = 20
flength = 0.5/n
supercell = [n, n, n]
thisdata = basedata.make_supercell(supercell)
center = np.array([0.5, 0.5-flength, 0.5])
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=center,
                                                  is_cartesian=False, delete=False, sort=True)

fcs = np.zeros(3, dtype=float)
nn = 0
for i in inds:
    thisdict = thisdata.atoms.iloc[i].to_dict()
    x = thisdict["xsn"] - center[0]
    y = thisdict["ysn"] - center[1]
    if x > flength/5.0 or abs(y) > flength/2.0:
        fcs[0] += thisdict["xsn"]
        fcs[1] += thisdict["ysn"]
        fcs[2] += thisdict["zsn"]
        nn += 1
fcs /= nn
cs = np.dot(thisdata.box.matrix.T, fcs)

for i in inds:
    thisdict = thisdata.atoms.iloc[i].to_dict()
    thiscoords = np.array([thisdict['x'], thisdict['y'], thisdict['z']])
    diff = cs - thiscoords
    d = np.linalg.norm(diff)

indict = {"type": 1, "x": cs[0], "y": cs[1], "z": cs[2]}
thisdata.add_an_entry(indict, loc=None, check_distance=False, rcut=rcut)
thisdata. assert_my_force_field()
thisdata.reset_atom_ids()
thisdata.to_file("SIA_Tet.dat")
print("finished SIA_Tet")
'''

'''
fname = "bcc.POSCAR"
basedata = lmpData.from_POSCAR(fname, atom_style)
b = 1.0
basedata.scale_data(b, style=0)
radius = 0.87 * b
rcut = 0.5 * b
supercell = [10, 10, 10]
thisdata = basedata.make_supercell(supercell)
inds, xyzs, ds, types = thisdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
thisdata.remove_by_inds(inds[0])
thisdata.reset_atom_ids()
thisdata.to_file("vacancy.dat")
print("finished vacancy")

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
'''
