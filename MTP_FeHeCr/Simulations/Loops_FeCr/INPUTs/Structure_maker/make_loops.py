import os,sys
import shutil
import copy
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices

from mylammps.utils.FeHeCr import lattice_parameters, ground_energies
import mylammps.utils.FeHeCr as MTPutil

supercell = [20, 40, 30]
supercell4layer = [20, 1, 30] #y must be 1
radius = 15
fbasisid = 0 #0:(111) else  (100)
flayerid = 0 #0:(111) else  (100)
mergy_style = 0 #0: interstitial else vacancy
loop_shape = "square" #"circle" or "square" or "rectangle".
                      #if square, the lateral length defaults 2 * radius unless lengths is specified.
                      #rectangle must input lengths three dimensions, but y is always depressed

MTPutil.create_dislocation_loop_basis(supercell, supercell4layer, radius, center=[0.5, 0.5, 0.5], a0=2.83048847,
                                      fbasisid=fbasisid, flayerid=flayerid,
                                      mergy_style=mergy_style, loop_shape=loop_shape,
                                      splits=[0.0, -0.25, -0.25], lengths=None)
