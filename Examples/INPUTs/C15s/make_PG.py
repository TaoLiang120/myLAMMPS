import os
import numpy as np
import copy
from pymatgen.core.operations import SymmOp
from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpData
from mylammps.inputs.util import mat_lengths, mat_angles, generate_rotation_matrix

from pymatgen.core.periodic_table import Element
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Molecule, Structure

from pymatgen.core.operations import SymmOp
from pymatgen.symmetry.analyzer import (
    SpacegroupAnalyzer,
    PointGroupAnalyzer,
)

def to_molecule(thisdata, ff_elements):
    ff_elements = np.array(ff_elements)
    types = thisdata.atoms['type'].to_numpy().astype(int) - 1
    coords = np.vstack((thisdata.atoms['x'], thisdata.atoms['y'], thisdata.atoms['z']))
    coords = coords.T
    eles = ff_elements[types]
    return Molecule(eles, coords)

def get_PG_OPs(thisdata, ff_elements):
    thisMol = to_molecule(thisdata, ff_elements)
    PA = PointGroupAnalyzer(thisMol)
    thisPGOPs = PA.get_symmetry_operations()
    return thisPGOPs, PA.sch_symbol

def get_SG_OPs(thisdata):
    SA = SpacegroupAnalyzer(thisdata.to_structure())
    thisSGOPs = SA.get_symmetry_operations()
    return thisSGOPs

fname = "MgCu2.POSCAR"
thisdata = lmpData.from_POSCAR(fname, "atomic")
ff_elements = ["Mg", "Cu"]


thisPGOPs, sch_sym = get_PG_OPs(thisdata, ff_elements)

'''
thisSGOPs = get_SG_OPs(thisdata)
icount = 0
for iop in range(len(thisSGOPs)):
    symmop = thisSGOPs[iop]
    if np.linalg.norm(symmop.translation_vector - np.array([0.5, 0.25, 0.25])) < 5:
        icount += 1
        print(iop)
        print(symmop.rotation_matrix)
        print(symmop.translation_vector)
        print(icount)
        print("---")
'''
rot0 = [[1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]

rot1 = [[0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0]]


rot2 = [[0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]]

rot3 = [[0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]]

rots = [rot0, rot1, rot2, rot3]
origin = (0, 0, 0)

C15_sequences = [[0, 0, 0],
                 [1, 1, 1], [-1, -1, 1], [1, -1, -1], [-1, 1, -1],
                 [1, -1, 1], [-1, 1, 1], [1, 1, -1], [-1, -1, -1],
                 [0, 2, 2], [2, 0, 2], [2, 2, 0],
                 [0, 0, 2], [0, 2, 0], [2, 0, 0], [2, 2, 2],
                 ]

print(thisdata.atoms)
print("===")
for iop in range(len(thisPGOPs)):
    tmpdata = thisdata.deepcopy()
    #seq = C15_sequences[iseq]
    #symmop = SymmOp.from_rotation_and_translation(rot, origin)
    #print(rot)
    symmop = thisPGOPs[iop]
    rot_matrix = np.around(symmop.rotation_matrix, decimals=2)
    print(rot_matrix.tolist())

    #tmpdata.atoms = lmpData.modify_by_symmetry(tmpdata.atoms, symmop, normalization=True)
    #print(tmpdata.atoms)
    print("---")
