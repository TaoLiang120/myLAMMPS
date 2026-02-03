import numpy as np
import copy
import itertools
from numpy import pi
from mylammps.inputs.data import lmpData, lmpBox

a = 1.0
'''
lvac = 7.0*a
thres = [0.03*a, 0.03*a, 0.03*a]
fname = "bcc.POSCAR"
atom_style = "atomic"
basedata = lmpData.from_POSCAR(fname, atom_style)
n = 10
supercell = [n, n, n]
thisdata = basedata.make_supercell(supercell)
thisdata.add_vacuum(lvac=lvac, direction=2, zero_coords=True, thres=thres)
thisdata.to_file("Surf001.dat")
'''


a = 2.85
fname = "bcc110_112_111.data"
atom_style = "atomic"
basedata = lmpData.from_file(fname, atom_style)
supercell = [2, 3, 3]
thisdata = basedata.make_supercell(supercell)
outdata = thisdata.deepcopy()
outdata.scale_data(a, style=0)
#outdata.add_vacuum(lvac=lvac, direction=2, zero_coords=True, thres=thres)
newaxes = [0, 2, 1]
outdata.swap_axes(newaxes)
outdata.to_file("Surf211.dat")

'''
fname = "Surf211.dat"
atom_style = "atomic"
thisdata = lmpData.from_file(fname, atom_style)
thisdata.bcc112_to_omega(a, nz=1, zstart=0, zstart4shift=6, thres=[0.2, 0.2, 0.2])
thisdata.to_file("omega.dat")
'''


d112 = 0.408248 * a
fname = "Surf211.dat"
atom_style = "atomic"
thisdata = lmpData.from_file(fname, atom_style)
dshift = np.sqrt(3)/6*a
rshift = 1.5
zstart4shift=7
thisdata.GPSF_move(rshift, dshift, d112, zstart=0, zstart4shift=zstart4shift)
thisdata.to_POSCAR("GPSF112_" + str(rshift) + ".POSCAR")

d112 = 0.408248 * a
fname = "Surf211.dat"
atom_style = "atomic"
thisdata = lmpData.from_file(fname, atom_style)
dshift = np.sqrt(3)/6*a
rshift = 1.5
zstart4shift=8
thisdata.GSFE_move(rshift, dshift, d112, zstart=0, zstart4shift=zstart4shift)
thisdata.to_POSCAR("GSFE112_" + str(rshift) + ".POSCAR")

'''
a = 2.85
fname = "bcc110_112_111.data"
atom_style = "atomic"
basedata = lmpData.from_file(fname, atom_style)
supercell = [5, 1, 3]
thisdata = basedata.make_supercell(supercell)
outdata = thisdata.deepcopy()
outdata.scale_data(a, style=0)
#outdata.add_vacuum(lvac=lvac, direction=2, zero_coords=True, thres=thres)
newaxes = [1, 2, 0]
outdata.swap_axes(newaxes)
outdata.to_file("Surf110.dat")
'''

'''
d110 = np.sqrt(2) / 2 * a
fname = "Surf110.dat"
atom_style = "atomic"
thisdata = lmpData.from_file(fname, atom_style)
dshift = np.sqrt(3)/6*a
rshift = 1.5
thisdata.GPSF_move(rshift, dshift, d110, zstart=0, zstart4shift="auto")
thisdata.to_POSCAR("GPSF110_" + str(rshift) + ".POSCAR")

'''

'''
a = 2.85
fname = "bcc110_112_111.data"
atom_style = "atomic"
basedata = lmpData.from_file(fname, atom_style)
supercell = [4, 1, 3]
thisdata = basedata.make_supercell(supercell)
outdata = thisdata.deepcopy()
outdata.scale_data(a, style=0)
#outdata.add_vacuum(lvac=lvac, direction=2, zero_coords=True, thres=thres)
newaxes = [1, 2, 0]
outdata.swap_axes(newaxes)
outdata.to_file("Surf110.dat")

d110 = 0.707107 * a
fname = "Surf110.dat"
atom_style = "atomic"
thisdata = lmpData.from_file(fname, atom_style)
dshift = np.sqrt(3)/6*a
rshift = 0
thisdata.atoms = thisdata.atoms.sample(frac=1)
thisdata.GSFE_move(rshift, dshift, d110, zstart=0, zstart4shift="auto")
thisdata.to_POSCAR("GSFE_" + str(rshift) + ".POSCAR")
'''

'''
outdata = thisdata.deepcopy()
outdata.add_vacuum(lvac=lvac, direction=0, zero_coords=True, thres=thres)
newaxes = [2, 1, 0]
outdata.swap_axes(newaxes)
outdata.to_file("Surf110.dat")

outdata = thisdata.deepcopy()
outdata.add_vacuum(lvac=lvac, direction=1, zero_coords=True, thres=thres)
newaxes = [0, 2, 1]
outdata.swap_axes(newaxes)
outdata.to_file("Surf112.dat")
'''
