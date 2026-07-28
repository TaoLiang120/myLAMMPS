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
Cr_Concs = [0.03, 0.10]
a0Crs = [2.83218, 2.83511]
epaCrs =[-8.27616063, -8.35194938]
epaCrs_partition = [[-7.0433014175257735, -48.138606458333335], [-4.280546540747361, -44.994575187500004]]

a0=2.83048847

fbasis = "bcc.data"
fbasis_int = "tetra_bcc.data"
supercell = [20, 20, 20]
supercell4int = [4, 4, 4]

nHes = [2]
mVs = [1]
center0 = [0.5, 0.5, 0.5]

print("Making structures...")

outfolders = MTPutil.create_HenVms_from_basis(supercell, fbasis, nHes, mVs,
                                              fbasis_int=fbasis_int, a0=a0,
                                              supercell4int=supercell4int,
                                              center=center0, is_cartesian=False, style=0,
                                              refdata=None, outfheader="Fe")

print("outfolders: ", outfolders, "")