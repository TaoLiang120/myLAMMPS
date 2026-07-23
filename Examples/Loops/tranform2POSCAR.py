import os, sys
import numpy as np
import copy
import itertools
from numpy import pi

from mylammps.inputs.data import lmpData

ff_elements = ["Fe", "He"]
atomic_masses = [55.845, 4.000]

a0 = 2.8553127490030867
fname = "VL111_r4.845.dat"
outfname = "in.dat"
basedata = lmpData.from_file(fname, atom_style="atomic")
basedata.scale_data(a0)
basedata.assert_force_field(ff_elements, atomic_masses)
basedata.to_file(outfname)