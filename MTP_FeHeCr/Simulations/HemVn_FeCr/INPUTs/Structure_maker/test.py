import os,sys
import shutil
import copy
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices

from mylammps.utils.FeHeCr import lattice_parameters, ground_energies
import mylammps.utils.FeHeCr as MTPutil

ff_elements = ["Fe", "He", "Cr"]
atomic_masses = [55.845, 4.0026, 51.9961]
lattice_parameters = [2.83048847, 2.67054960, 2.82958219]
ground_energies = [-8.24159650, 0.01474557, -9.45576250]
Cr_Concs = [0.03, 0.09]

supercell = [10, 10, 10]
fbasis = "bcc.data"

a0=2.83048847

outdata = lmpData.from_file(fbasis, "atomic", sort_id=False, parse_velocity=False)
outdata = outdata.make_supercell(supercell)
outdata.scale_data(a0, style=0)
outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)

inds, xyzs, ds, types = outdata.select_by_radius(2.85, depress=None, center=[0.5, 0.5, 0.5],
                                                     is_cartesian=False, style=0,
                                                     delete=False, sort=True)

#inds_sort = MTPutil.sort_by_dynamic_center(xyzs)

inds_sort=MTPutil.sort_by_ref_coords(xyzs, xyzs[::-1])
print(inds_sort)


