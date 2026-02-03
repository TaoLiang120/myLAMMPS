import copy
import sys
import itertools
import re
from io import StringIO

import numpy as np
import pandas as pd
from monty.json import MSONable
from numpy import pi
from pymatgen.core.lattice import Lattice
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.lammps.data import ATOMS_HEADERS
from pymatgen.io.lammps.data import LammpsBox, LammpsData, ForceField, Topology

def standardize_lattice_matrix(lattice, origin=(0.0, 0.0, 0.0)):
    a, b, c = lattice.abc
    xlo, ylo, zlo = origin
    xhi = a + xlo
    m = lattice.matrix
    xy = np.dot(m[1], m[0] / a)
    yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
    xz = np.dot(m[2], m[0] / a)
    yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
    zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo

    newmatrix = [[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]]
    rot_matrix = np.linalg.solve(newmatrix, m)
    symmop = SymmOp.from_rotation_and_translation(rot_matrix, origin)
    return newmatrix, symmop

fname = "test.POSCAR"
thisstr = Structure.from_file(fname)
print(thisstr.lattice.matrix)
print(thisstr.cart_coords)
frac_coords = [site.frac_coords for site in thisstr.sites]
print(frac_coords)
print("000000")

newmatrix, symmop = standardize_lattice_matrix(thisstr.lattice)

coords = symmop.operate_multi(thisstr.cart_coords)
print(thisstr.cart_coords)
print(coords)
print("11111111")


coords = symmop.operate_multi(thisstr.cart_coords)
print(thisstr.cart_coords)
print(coords)
print("22222222222222222222")
newlatt = Lattice(newmatrix)
for i in range(len(thisstr.sites)-1, -1, -1):
    site = thisstr.sites[i]
    cs = site.coords
    fcs = site.frac_coords
    
    print(f"before cs:{cs} fcs:{fcs}")
    print(f"cart_coords: {thisstr.cart_coords[i]}")
    cs_after = symmop.operate_multi(cs)
    fcs_after = np.dot(cs_after, newlatt.inv_matrix)
    print(f"after cs:{cs} fcs:{fcs} cs_after:{cs_after} fcs_after:{fcs_after}")
    print("---")
print("========")
thisstr.lattice = Lattice(newmatrix)
print(thisstr.cart_coords)
frac_coords = [site.frac_coords for site in thisstr.sites]
print(frac_coords)
#coords = symmop.operate_multi(thisstr.cart_coords)
print("finished")


compstr = "HFeCrHe"
comp = Composition(compstr)
print(comp.reduced_formula)

myElements = ["Cr", "Fe", "H", "He"]

for i in range(len(myElements)):
    s = myElements[i]
    e = Element(s)
  
    r = e.atomic_radius
    if not isinstance(r, float): 
        r = e.atomic_radius_calculated
    v = 4.0 / 3.0 * pi * np.power(r, 3)
    print(s, r, v)
