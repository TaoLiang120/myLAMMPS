import os,sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData

fname = "bcc112_111_110_orth4edge.data"
atom_style = "atomic"
force_field = ["Fe"]

a = 2.85
lvac = 7.0*a
thres = [0.03*a, 0.03*a, 0.03*a]
burgerm = a
thisdata = lmpData.from_file(fname, atom_style, sort_id=False, parse_velocity=False)
thisdata.scale_data(a, style=0)
supercell = [14, 18, 22]
outdata=thisdata.make_supercell(supercell)

outdata.create_edge_dislocation(burgerm, nedges=1, add_vacuum=True, direction=2, lvac=lvac)
outdata.to_file("Edge111.dat")
