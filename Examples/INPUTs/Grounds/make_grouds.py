import os,sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData

fname = "bcc.POSCAR"
atom_style = "atomic"
is_sort = False
force_field = ["Fe"]

a = 2.85    

thisdata = lmpData.from_POSCAR(fname, atom_style, is_sort=is_sort)
supercell = [10, 10, 10]
outdata=thisdata.make_supercell(supercell)

outdata.assert_force_field(force_field)
#outdata.scale_data(a, style=0)
outdata.to_file(outdata.full_formula + ".dat")

