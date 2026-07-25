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


outfiles = ["Ref_Fe.dat", "Ref_Fe_3Cr.dat", "Ref_Fe_10Cr.dat"]
outheaders = ["Fe", "Fe_3Cr", "Fe_10Cr"]


supercell = [5, 5, 5]
supercell4layer = [50, 1, 50] #y must be 1
radius = 15
fbasis = "bcc_10x10x10.data"
fbasisid = 1 #0:(111) else  (100)
flayerid = 1 #0:(111) else  (100)
mergy_style = 0 #0: interstitial else vacancy
loop_shape = "square" #"circle" or "square" or "rectangle".
                      #if square, the lateral length defaults 2 * radius unless lengths is specified.
                      #rectangle must input lengths three dimensions, but y is always depressed

outdata = MTPutil.create_dislocation_loop_basis(supercell, supercell4layer, radius,
                                      center=[0.5, 0.5, 0.5], a0=2.83048847,
                                      fbasis=fbasis, fbasisid=fbasisid, flayerid=flayerid,
                                      mergy_style=mergy_style, loop_shape=loop_shape,
                                      splits=[0.75, 1.3, 0.0], lengths=None)

outdata.atoms['type'] = [1] * len(outdata.atoms)
ff_elements = ["Fe", "He"]
atomic_masses = [55.845, 4.0026]
outdata.assert_force_field(ff_elements, atomic_masses)
outfile = "I100_s30_Fe.dat"
outdata.to_file(outfile)
