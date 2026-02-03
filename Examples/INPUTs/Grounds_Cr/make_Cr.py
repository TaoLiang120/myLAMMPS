import os,sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData
from mylammps.utils.FeHHeCr import get_data_files, set_2to3, Fe2Cr, vasp2lmp
from mylammps.utils.FeHHeCr import Fe2Crs_from_a_file


fname = "bcc.POSCAR"
atom_style = "atomic"
is_sort = False

myElements = ["Fe", "H", "He", "Cr"]
myAtomicMasses = [55.845, 1.008, 4.003, 51.996]

a = 2.85

Cr_concs = [0.04, 1.0, 1.0, 1.0]
to_typeids = [4, 4, 2, 3]
thisdata = lmpData.from_POSCAR(fname, atom_style, is_sort=is_sort)
supercell = [10, 10, 10]
orgdata=thisdata.make_supercell(supercell)

orgdata.assert_force_field(myElements, myAtomicMasses)
#outdata.scale_data(a, style=0)
#outdata.to_file(outdata.full_formula + ".dat")

'''
for i in range(len(Cr_concs)):
   outdata = orgdata.deepcopy()
   c = Cr_concs[i]
   outdata = Fe2Cr(outdata, c, to_typeid=to_typeids[i])
   #outdata.scale_data(a, style=0)
   outdata.get_data_info()
   outdata.to_file(outdata.full_formula + ".dat")
'''

fname = "H2000.dat"
thisdata = lmpData.from_file(fname, atom_style)
print(thisdata.full_formula)