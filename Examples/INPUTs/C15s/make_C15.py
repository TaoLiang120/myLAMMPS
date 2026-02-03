import os
import numpy as np
import copy
import pandas as pd
from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpData

fname = "bcc.POSCAR"
bccdata = lmpData.from_POSCAR(fname, "atomic")
lattpara = 2.83037145
lattpara = 2.86
ff_elements=["Fe", "Cu"]
atomic_masses=[55.845, 64.0]
bccdata = bccdata.make_supercell_simple([12, 12, 12])
bccdata.scale_data(lattpara, style=0)
typesoffset = True
nsias = 128
outdata = bccdata.create_C15_BCC(lattpara, nsias, to_center=[0.5, 0.5, 0.5], is_cartesian=False,
                       fC15_0="Z12_0.POSCAR", fC15_1="Z12_1.POSCAR", atom_style="atomic",
                       typesoffset=typesoffset, ff_elements=ff_elements, atomic_masses=atomic_masses)
outdata.to_file("C15_I"+str(nsias)+".dat")
#outdata.to_POSCAR("C15_I"+str(nsias)+".POSCAR")
