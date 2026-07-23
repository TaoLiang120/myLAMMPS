# coding: utf-8
# Copyright (c) Tao Liang.
# Distributed under the terms of the MIT License.

import copy
import sys
import itertools
import re
from io import StringIO

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from monty.json import MSONable
from numpy import pi
from pymatgen.core.lattice import Lattice
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.core.operations import SymmOp
from pymatgen.io.lammps.data import ATOMS_HEADERS
from pymatgen.io.lammps.data import LammpsBox, LammpsData, ForceField, Topology
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.util.io_utils import clean_lines

from mylammps.myglobal import myElements, myAtomicVolumes, myAtomicMasses, myAtomicNumbers
from mylammps.elastic.distortion import Distortion
from mylammps.inputs.util import mat_lengths, mat_angles, generate_rotation_matrix

__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

SECTION_KEYWORDS = {
    "atom": [
        "Atoms",
        "Velocities",
        "Masses",
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
    ],
    "topology": ["Bonds", "Angles", "Dihedrals", "Impropers"],
    "ff": [
        "Pair Coeffs",
        "PairIJ Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
    ],
    "class2": [
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ],
}

CLASS2_KEYWORDS = {
    "Angle Coeffs": ["BondBond Coeffs", "BondAngle Coeffs"],
    "Dihedral Coeffs": [
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
    ],
    "Improper Coeffs": ["AngleAngle Coeffs"],
}

SECTION_HEADERS = {
    "Masses": ["mass"],
    "Velocities": ["vx", "vy", "vz"],
    "Bonds": ["type", "atom1", "atom2"],
    "Angles": ["type", "atom1", "atom2", "atom3"],
    "Dihedrals": ["type", "atom1", "atom2", "atom3", "atom4"],
    "Impropers": ["type", "atom1", "atom2", "atom3", "atom4"],
}

ATOMS_HEADERS = {
    "angle": ["molecule-ID", "type", "x", "y", "z"],
    "atomic": ["type", "x", "y", "z"],
    "bond": ["molecule-ID", "type", "x", "y", "z"],
    "charge": ["type", "q", "x", "y", "z"],
    "full": ["molecule-ID", "type", "q", "x", "y", "z"],
    "molecular": ["molecule-ID", "type", "x", "y", "z"],
}


class lmpBox(LammpsBox, MSONable):
    def __init__(
            self,
            bounds,
            tilt=None,
    ):
        if not tilt: tilt = np.zeros(3, dtype=float)

        super().__init__(bounds, tilt)
        #self._matrix = matrix  # type: np.ndarray
        self._inv_matrix = None  # type: Optional[np.ndarray]

    @property
    def lengths(self) -> Tuple[float, float, float]:
        """
        :return: The lengths (a, b, c) of the lattice.
        """
        return tuple(mat_lengths(self._matrix).tolist())  # type: ignore

    @property
    def origin(self) -> Tuple[float, float, float]:
        return tuple([self.bounds[0][0], self.bounds[1][0], self.bounds[2][0]])

    @property
    def angles(self) -> Tuple[float, float, float]:
        return tuple(mat_angles(self._matrix).tolist())  # type: ignore

    @property
    def is_orthogonal(self) -> bool:
        """
        :return: Whether all angles are 90 degrees.
        """
        return all(abs(a - 90) < 1e-5 for a in self.angles)

    def copy(self):
        """Deep copy of self."""
        return self.__class__(self.matrix.copy())

    @property
    def matrix(self) -> np.ndarray:
        """Copy of matrix representing the Lattice"""
        return self._matrix

    @property
    def inv_matrix(self) -> np.ndarray:
        """
        Inverse of lattice matrix.
        """
        if self._inv_matrix is None:
            self._inv_matrix = np.linalg.inv(self._matrix)
            self._inv_matrix.setflags(write=False)
        return self._inv_matrix


def lattice_2_lmpbox(lattice, origin=(0, 0, 0)):
    a, b, c = lattice.abc
    xlo, ylo, zlo = origin
    xhi = a + xlo
    m = lattice.matrix
    xy = np.dot(m[1], m[0] / a)
    yhi = np.sqrt(b ** 2 - xy ** 2) + ylo
    xz = np.dot(m[2], m[0] / a)
    yz = (np.dot(m[1], m[2]) - xy * xz) / (yhi - ylo)
    zhi = np.sqrt(c ** 2 - xz ** 2 - yz ** 2) + zlo
    tilt = None if lattice.is_orthogonal else [xy, xz, yz]
    bounds = [[xlo, xhi], [ylo, yhi], [zlo, zhi]]
    rot_matrix = np.linalg.solve([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]], m)
    symmop = SymmOp.from_rotation_and_translation(rot_matrix, origin)
    return lmpBox(bounds, tilt), symmop


def find_symmop_lattices(target, lattice):
    translation = (0, 0, 0)
    a1, b1, c1 = target.abc
    a2, b2, c2 = lattice.abc
    m1 = target.matrix
    m2 = copy.deepcopy(lattice.matrix)
    r1 = a2 / a1
    r2 = b2 / b1
    r3 = c2 / c1
    m2[0] /= r1
    m2[1] /= r2
    m2[2] /= r3
    rot_matrix = np.linalg.solve(m2, m1)
    symmop = SymmOp.from_rotation_and_translation(rot_matrix, translation)
    return symmop


class lmpData(LammpsData, MSONable):
    def __init__(
            self,
            box,
            masses,
            atoms,
            velocities=None,
            force_field=None,
            topology=None,
            atom_style="full",
    ):
        box = lmpBox(box.bounds, box.tilt)

        ##atoms = atoms[ATOMS_HEADERS[atom_style]]
        super().__init__(
            box,
            masses,
            atoms,
            velocities=velocities,
            force_field=force_field,
            topology=topology,
            atom_style=atom_style)

        if self.atoms.index.has_duplicates:
            print("The input data have duplicated atom ID.")
            sys.exit()
        ids = self.atoms.index.to_numpy().astype(int)
        self.idmax = np.max(ids)
        self.initialization(normalization=False, style=1)

    @classmethod
    def from_pmg_structure(cls, structure, atom_style, ff_elements=None, is_sort=False):
        if is_sort:
            s = structure.get_sorted_structure()
        else:
            s = structure.copy()
        box, symmop = lattice_2_lmpbox(s.lattice)
        #coords = symmop.operate_multi(s.cart_coords)
        cart_coords = [site.coords for site in structure.sites]
        coords = symmop.operate_multi(np.array(cart_coords))
        site_properties = s.site_properties
        if "velocities" in site_properties:
            velos = np.array(s.site_properties["velocities"])
            rot = SymmOp.from_rotation_and_translation(symmop.rotation_matrix)
            rot_velos = rot.operate_multi(velos)
            site_properties.update({"velocities": rot_velos})
        boxed_s = Structure(
            box.to_lattice(),
            s.species,
            coords,
            site_properties=site_properties,
            coords_are_cartesian=True,
        )

        symbols = list(s.symbol_set)
        if ff_elements:
            symbols.extend(ff_elements)
        elements = sorted(Element(el) for el in set(symbols))
        mass_info = [tuple([i.symbol] * 2) for i in elements]
        ff = ForceField(mass_info)
        topo = Topology(boxed_s)
        return cls.from_ff_and_topologies(box=box, ff=ff, topologies=[topo], atom_style=atom_style)

    @classmethod
    def from_POSCAR(cls, filename, atom_style, ff_elements=None, is_sort=False):
        s = Structure.from_file(filename)
        return cls.from_pmg_structure(s, atom_style, ff_elements=ff_elements, is_sort=is_sort)

    @classmethod
    def from_file(cls, filename, atom_style, sort_id=False, parse_velocity=False):
        """
        Constructor that parses a file.

        Args:
            filename (str): Filename to read.
            atom_style (str): Associated atom_style. Default to "full".
            sort_id (bool): Whether sort each section by id. Default to
                True.

        """
        with open(filename) as f:
            lines = f.readlines()
        kw_pattern = r"|".join(itertools.chain(*SECTION_KEYWORDS.values()))
        section_marks = [i for i, l in enumerate(lines) if re.search(kw_pattern, l)]
        parts = np.split(lines, section_marks)

        float_group = r"([0-9eE.+-]+)"
        header_pattern = {}
        header_pattern["counts"] = r"^\s*(\d+)\s+([a-zA-Z]+)$"
        header_pattern["types"] = r"^\s*(\d+)\s+([a-zA-Z]+)\s+types$"
        header_pattern["bounds"] = r"^\s*{}$".format(r"\s+".join([float_group] * 2 + [r"([xyz])lo \3hi"]))
        header_pattern["tilt"] = r"^\s*{}$".format(r"\s+".join([float_group] * 3 + ["xy xz yz"]))

        header = {"counts": {}, "types": {}}
        bounds = {}
        for l in clean_lines(parts[0][1:]):  # skip the 1st line
            match = None
            for k, v in header_pattern.items():
                match = re.match(v, l)
                if match:
                    break
            if match and k in ["counts", "types"]:
                header[k][match.group(2)] = int(match.group(1))
            elif match and k == "bounds":
                g = match.groups()
                bounds[g[2]] = [float(i) for i in g[:2]]
            elif match and k == "tilt":
                header["tilt"] = [float(i) for i in match.groups()]
        header["bounds"] = [bounds.get(i, [-0.5, 0.5]) for i in "xyz"]
        box = lmpBox(header["bounds"], header.get("tilt"))

        def parse_section(sec_lines):
            title_info = sec_lines[0].split("#", 1)
            kw = title_info[0].strip()
            sio = StringIO("".join(sec_lines[2:]))  # skip the 2nd line
            if kw.endswith("Coeffs") and not kw.startswith("PairIJ"):
                df_list = [
                    pd.read_csv(StringIO(line), header=None, comment="#", sep='\s+')
                    for line in sec_lines[2:]
                    if line.strip()
                ]
                df = pd.concat(df_list, ignore_index=True)
                names = ["id"] + ["coeff%d" % i for i in range(1, df.shape[1])]
            else:
                df = pd.read_csv(sio, header=None, comment="#", sep='\s+')
                if kw == "PairIJ Coeffs":
                    names = ["id1", "id2"] + ["coeff%d" % i for i in range(1, df.shape[1] - 1)]
                    df.index.name = None  # pylint: disable=E1101
                elif kw in SECTION_HEADERS:
                    names = ["id"] + SECTION_HEADERS[kw]
                elif kw == "Atoms":
                    names = ["id"] + ATOMS_HEADERS[atom_style]
                    if df.shape[1] == len(names):  # pylint: disable=E1101
                        pass
                    elif df.shape[1] == len(names) + 3:  # pylint: disable=E1101
                        names += ["nx", "ny", "nz"]
                    else:
                        raise ValueError("Format in Atoms section inconsistent with atom_style %s" % atom_style)
                else:
                    raise NotImplementedError("Parser for %s section not implemented" % kw)
            df.columns = names
            if sort_id:
                sort_by = "id" if kw != "PairIJ Coeffs" else ["id1", "id2"]
                df.sort_values(sort_by, inplace=True)
            if "id" in df.columns:
                df.set_index("id", drop=True, inplace=True)
                df.index.name = None
            return kw, df

        err_msg = "Bad LAMMPS data format where "
        body = {}
        seen_atoms = False
        for part in parts[1:]:
            name, section = parse_section(part)
            if name == "Atoms":
                seen_atoms = True
            if (
                    name in ["Velocities"] + SECTION_KEYWORDS["topology"] and not seen_atoms
            ):  # Atoms must appear earlier than these
                raise RuntimeError(err_msg + "%s section appears before Atoms section" % name)
            body.update({name: section})

        err_msg += "Nos. of {} do not match between header and {} section"
        assert len(body["Masses"]) == header["types"]["atom"], err_msg.format("atom types", "Masses")
        atom_sections = ["Atoms", "Velocities"] if "Velocities" in body else ["Atoms"]
        for s in atom_sections:
            assert len(body[s]) == header["counts"]["atoms"], err_msg.format("atoms", s)
        for s in SECTION_KEYWORDS["topology"]:
            if header["counts"].get(s.lower(), 0) > 0:
                assert len(body[s]) == header["counts"][s.lower()], err_msg.format(s.lower(), s)

        items = {k.lower(): body[k] for k in ["Masses", "Atoms"]}
        if parse_velocity:
            items["velocities"] = body.get("Velocities")
        else:
            items["velocities"] = None
        ff_kws = [k for k in body if k in SECTION_KEYWORDS["ff"] + SECTION_KEYWORDS["class2"]]
        items["force_field"] = {k: body[k] for k in ff_kws} if ff_kws else None
        topo_kws = [k for k in body if k in SECTION_KEYWORDS["topology"]]
        items["topology"] = {k: body[k] for k in topo_kws} if topo_kws else None
        items["atom_style"] = atom_style
        items["box"] = box
        return cls(**items)

    def to_atom_style(self):
        self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]

    def to_file(self, filename, to_atom_style=True):
        if to_atom_style: self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]
        self.write_file(filename)

    def to_structure(self):
        self.atoms = self.atoms[ATOMS_HEADERS[self.atom_style]]
        return self.structure

    def to_POSCAR(self, filename, direct=True, significant_figures=18):
        s = self.to_structure()
        p = Poscar(s)
        p.write_file(filename, direct=direct, significant_figures=significant_figures)

    def deepcopy(self):
        tmpdata = copy.deepcopy(self)
        return tmpdata

    def remove_atoms_tag(self, tag):
        if tag in self.atoms.columns:
            self.atoms = self.atoms.drop([tag], axis=1)
        if tag == 'molecule-ID':
            if self.atom_style == 'molecular':
                self.atom_style = 'atomic'
            elif self.atom_style == 'full':
                self.atom_style = 'charge'

    def insert_atoms_tag(self, tag, val):
        if isinstance(val, list) or isinstance(val, tuple) or isinstance(val, np.ndarray):
            pass
        else:
            val = [val] * self.natoms
        self.remove_atoms_tag(tag)
        self.atoms.insert(len(self.atoms.columns), tag, val)

    def insert_itags(self):
        itags = np.arange(len(self.atoms), dtype=int)
        self.insert_atoms_tag("itag", itags)

    def insert_molecular_id(self, mol_id=0):
        if self.atom_style == 'atomic':
            self.atom_style = 'molecular'
            im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
            self.atoms.insert(im, 'molecule-ID', np.array([mol_id] * self.natoms, dtype=int))
        elif self.atom_style == 'charge':
            self.atom_style = 'full'
            im = ATOMS_HEADERS[self.atom_style].index('molecule-ID')
            self.atoms.insert(im, 'molecule-ID', np.array([mol_id] * self.natoms, dtype=int))

    def remove_molecular_id(self):
        self.remove_atoms_tag("molecule-ID")
        if self.atom_style == 'full':
            self.atom_style = 'charge'
        elif self.atom_style == 'molecular':
            self.atom_style = 'atomic'

    def assert_force_field(self, ff_elements, atomic_masses=None):
        force_field = {}
        for i in range(len(ff_elements)):
            force_field[str(i + 1)] = ff_elements[i]

        self.force_field = force_field
        if atomic_masses is None:
            atomic_masses = copy.deepcopy(myAtomicMasses)
        masses = pd.DataFrame(atomic_masses, columns=["mass"], index=np.arange(len(atomic_masses), dtype=int) + 1)
        self.masses = masses
        self.get_data_info()

    def assert_my_force_field(self):
        self.assert_force_field(myElements, atomic_masses=myAtomicMasses)

    @staticmethod
    def normalize_frac_coords(a, style=1):
        a = np.array(a)
        a = np.subtract(a, a.round())
        if style == 1:
            a = np.select([a < 0, a <1.0, a>=1.0], [a + 1.0, a, a - np.floor(a)])
        return a

    @staticmethod
    def find_thermal_expansion(symbols, iatoms, temp):
        iatoms = np.array(iatoms)
        natoms = np.sum(iatoms)
        fatoms = iatoms / float(natoms)
        texp = 0.0
        for i in range(len(symbols)):
            el = Element(symbols[i])
            f = fatoms[i]
            try:
                e = el.__getattr__("coefficient_of_linear_thermal_expansion")
            except:
                e = 0.0
            texp += e * temp * f
        return texp

    @staticmethod
    def compute_site_distance(coords, tmpdata, rcut=1.0, style=0, sort=False):
        rcut2 = rcut * rcut
        inds = np.arange(tmpdata.natoms, dtype=int)
        types = tmpdata.atoms["type"].to_numpy().astype(int)
        if style == 0:
            thisxyzns = np.vstack((tmpdata.atoms["xsn"], tmpdata.atoms["ysn"], tmpdata.atoms["zsn"]))
            thiscoords = np.dot(coords, tmpdata.box.inv_matrix)
            xyzns = thisxyzns.T - thiscoords
            xyzns = xyzns.T
            xyzns[0] = lmpData.normalize_frac_coords(xyzns[0], style=0)
            xyzns[1] = lmpData.normalize_frac_coords(xyzns[1], style=0)
            xyzns[2] = lmpData.normalize_frac_coords(xyzns[2], style=0)
            thisxyznsq = np.sum(np.dot(xyzns.T, tmpdata.box.matrix) ** 2, axis=1)
        else:
            thisxyzns = np.vstack((tmpdata.atoms["x"], tmpdata.atoms["y"], tmpdata.atoms["z"]))
            xyzns = thisxyzns.T - coords
            thisxyznsq = np.sum(xyzns ** 2, axis=1)
        thisinds = np.compress(thisxyznsq < rcut2, inds, axis=0)
        thisxyzs = (thisxyzns.T)[thisinds]
        if style == 0:
            thisxyzs = np.dot(thisxyzs, tmpdata.box.matrix)
        thisxyznsq = thisxyznsq[thisinds]
        thisxyznsq = np.sqrt(thisxyznsq)
        types = types[thisinds]

        if sort:
            isorts = np.argsort(thisxyznsq)
            thisinds = thisinds[isorts]
            thisxyzs = thisxyzs[isorts]
            thisxyznsq = thisxyznsq[isorts]
            types = types[isorts]
        return thisinds, thisxyzs, thisxyznsq, types

    def coords2fracts(self, From_Cart=True, normalization=False, style=1, axes=[0, 1, 2]):
        if From_Cart:
            cols = ["x", "y", "z"]
            cart_coords = np.vstack((self.atoms[cols[axes[0]]], self.atoms[cols[axes[1]]], self.atoms[cols[axes[2]]]))
            frac_coords = np.dot(self.box.inv_matrix.T, cart_coords)
        else:
            cols = ["xsn", "ysn", "zsn"]
            frac_coords = np.vstack((self.atoms[cols[axes[0]]], self.atoms[cols[axes[1]]], self.atoms[cols[axes[2]]]))

        if normalization:
            for idim in range(3):
                frac_coords[idim] = lmpData.normalize_frac_coords(frac_coords[idim], style=style)

        dropcols = []
        if "xsn" in self.atoms.columns: dropcols.append("xsn")
        if "ysn" in self.atoms.columns: dropcols.append("ysn")
        if "zsn" in self.atoms.columns: dropcols.append("zsn")
        if len(dropcols) > 0: self.atoms = self.atoms.drop(dropcols, axis=1)
        self.atoms.insert(len(self.atoms.columns), "xsn", frac_coords[0])
        self.atoms.insert(len(self.atoms.columns), "ysn", frac_coords[1])
        self.atoms.insert(len(self.atoms.columns), "zsn", frac_coords[2])

    def fracts2coords(self):
        if "xsn" not in self.atoms.columns: self.coords2fracts(normalization=False)
        xyzns = np.vstack((self.atoms["xsn"], self.atoms["ysn"], self.atoms["zsn"]))
        xyzs = np.dot(self.box.matrix.T, xyzns)
        self.atoms["x"] = xyzs[0]
        self.atoms["y"] = xyzs[1]
        self.atoms["z"] = xyzs[2]

    def normalize_coords(self, style=1):
        self.coords2fracts(normalization=True, style=style)
        self.fracts2coords()

    def reset_atom_ids(self):
        self.atoms = self.atoms.reset_index(drop=True)
        self.atoms = self.atoms.set_index(np.arange(self.natoms, dtype=int) + 1)

    def find_cubic_lattice_parameter(self, myElements=myElements, myAtomicVolumes=myAtomicVolumes):
        vtot = 0
        for i in range(len(self.symbols)):
            sym = self.symbols[i]
            n = self.nsites_symbols[i]
            j = myElements.index(sym)
            v = myAtomicVolumes[j]
            vtot += v * n
        a = np.power(vtot, 1.0 / 3.0)
        return a

    def get_data_info(self):
        thistypes = self.atoms['type'].to_numpy()

        self.types, counts = np.unique(thistypes, return_counts=True)
        ntypes = np.max(self.types)
        if self.force_field is None:
            if self.masses is None:
                self.symbols = np.arange(ntypes, dtype=str) + 1
            else:
                self.symbols = self.masses.index.to_numpy().astype(str)
            for i in self.types:
                self.symbols[i - 1] = myElements[i - 1]
        else:
            self.symbols = ["0"] * len(self.force_field)
            for k in self.force_field:
                i = int(k)
                if i > len(self.symbols):
                    raise ValueError("Number of types is not equal to number of elements in force field.")
                self.symbols[i - 1] = self.force_field[k]

        if ntypes > len(self.symbols):
            raise ValueError("Number of types is larger than the number of elements in force field.")

        self.ntypes = max(ntypes, len(self.symbols))
        self.full_formula = ""
        self.nsites_symbols = [0] * len(self.symbols)
        for i in range(len(self.types)):
            t = self.types[i]
            sym = self.symbols[t - 1]
            c = counts[i]
            self.nsites_symbols[t - 1] = c
            if c >= 1000000:
                cs = str(int(c / 1000000)) + "M"
            elif c >= 10000:
                cs = str(int(c / 1000)) + "K"
            else:
                cs = str(c)
            if i == len(self.types) - 1:
                self.full_formula += sym + cs
            else:
                self.full_formula += sym + cs + "_"

    def integerization(self):
        self.atoms['type'] = self.atoms['type'].astype(int)
        if "molecule-ID" in  self.atoms.columns:
            self.atoms['molecule-ID'] = self.atoms['molecule-ID'].astype(int)


    def initialization(self, normalization=False, style=1):
        self.integerization()
        self.insert_itags()
        self.coords2fracts(normalization=normalization, style=style)
        self.natoms = len(self.atoms)
        self.get_data_info()
        self.itagmax = self.natoms - 1

    def generate_default_dict(self):
        cols = self.atoms.columns.tolist()
        default_dict = {}
        for key in cols:
            if key == "molecule-ID" or key == "type" or key == "tag" or key == "id" or key == "itag":
                if key == "itag":
                    default_dict[key] = self.natoms
                else:
                    default_dict[key] = 0
            else:
                default_dict[key] = 0.0
        return default_dict

    def remove_by_inds(self, inds, style="dyn"):
        if isinstance(inds, int) or isinstance(inds, np.int_):
            inds = [inds]
        inds = np.array(inds)
        if style == "index":
            inds_org = self.atoms.index.to_numpy().astype(int)
            goods = np.delete(inds_org, inds)
            self.atoms = self.atoms.loc[goods]
        else:
            inds_org = np.arange(self.natoms, dtype=int)
            goods = np.delete(inds_org, inds)
            self.atoms = self.atoms.iloc[goods]

        if len(self.atoms) > 0:
            self.initialization()

    def select_by(self, by="type", vlim=[0, 1], style="INCLUDE", delete=False):
        self.insert_itags()
        t = self.deepcopy()
        t.atoms = t.atoms[
            (t.atoms[by] >= vlim[0]) & (t.atoms[by] < vlim[1])]
        inds = t.atoms["itag"].to_numpy().astype(int)
        if style[0:3].upper() == "INC":
            if delete:
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            return inds
        else:
            inds_org = self.atoms["itag"].to_numpy().astype(int)
            inds = np.delete(inds_org, inds)
            if delete:
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            return inds

    def select_by_coords(self, xlim=[0.25, 0.75], ylim=[0.25, 0.75], zlim=[0.25, 0.75],
                         Fractional=True, style="INCLUDE", delete=False):
        self.insert_itags()
        t = self.deepcopy()
        if Fractional:
            t.atoms = t.atoms[
                (t.atoms["xsn"] >= xlim[0]) & (t.atoms["xsn"] < xlim[1]) & (t.atoms["ysn"] >= ylim[0]) & (
                        t.atoms["ysn"] < ylim[1]) & (t.atoms["zsn"] >= zlim[0]) & (
                        t.atoms["zsn"] < zlim[1])]
        else:
            t.atoms = t.atoms[
                (t.atoms["x"] >= xlim[0]) & (t.atoms["x"] < xlim[1]) & (t.atoms["y"] >= ylim[0]) & (
                        t.atoms["y"] < ylim[1]) & (t.atoms["z"] >= zlim[0]) & (t.atoms["z"] < zlim[1])]

        inds = t.atoms["itag"].to_numpy().astype(int)
        if style[0:3].upper() == "INC":
            if delete:
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            return inds
        else:
            inds_org = self.atoms["itag"].to_numpy().astype(int)
            inds = np.delete(inds_org, inds)
            if delete:
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            return inds

    def shuffle_atoms(self, reset_ids=True, reset_itags=True):
        self.atoms = self.atoms.sample(frac=1)
        if reset_ids:
            self.reset_atom_ids()
        if reset_itags:
            self.insert_itags()
        self.get_data_info()

    def select_by_radius(self, radius, center=[0.5, 0.5, 0.5], is_cartesian=False, depress=1,
                         style=0, delete=False, delete_style="OUTSIDE", sort=False):
        self.insert_itags()
        radius2 = radius * radius
        center = np.array(center)
        inds = np.arange(self.natoms, dtype=int)
        thistypes = self.atoms['type'].to_numpy()
        if style == 0:
            if is_cartesian:
                center = np.dot(center, self.box.inv_matrix)
            org_xyzns = np.vstack((self.atoms["xsn"], self.atoms["ysn"], self.atoms["zsn"]))
            thisxyzns = copy.deepcopy(org_xyzns)
            if isinstance(depress, int):
                thisxyzns[depress] = np.zeros(len(thisxyzns[depress]))
                center[depress] = 0.0
            xyzns = thisxyzns.T - center
            xyzns = xyzns.T
            xyzns[0] = lmpData.normalize_frac_coords(xyzns[0], style=0)
            xyzns[1] = lmpData.normalize_frac_coords(xyzns[1], style=0)
            xyzns[2] = lmpData.normalize_frac_coords(xyzns[2], style=0)
            thisxyznsq = np.sum(np.dot(xyzns.T, self.box.matrix) ** 2, axis=1)
        else:
            org_xyzns = np.vstack((self.atoms["x"], self.atoms["y"], self.atoms["z"]))
            thisxyzns = copy.deepcopy(org_xyzns)
            if not is_cartesian:
                center = np.dot(center, self.box.matrix)
            if isinstance(depress, int):
                thisxyzns[depress] = np.zeros(len(thisxyzns[depress]))
                center[depress] = 0.0
            xyzns = thisxyzns.T - center
            thisxyznsq = np.sum(xyzns ** 2, axis=1)

        inds = np.compress(thisxyznsq < radius2, inds, axis=0)
        org_xyzns = (org_xyzns.T)[inds]
        if style == 0:
            org_xyzns = np.dot(org_xyzns, self.box.matrix)
        thisxyznsq = thisxyznsq[inds]
        thisxyznsq = np.sqrt(thisxyznsq)
        thistypes = thistypes[inds]

        if sort:
            isorts = np.argsort(thisxyznsq)
            inds = inds[isorts]
            org_xyzns = org_xyzns[isorts]
            thisxyznsq = thisxyznsq[isorts]
            thistypes = thistypes[isorts]

        if delete:
            if delete_style[0:3].upper() == "OUT":
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            else:
                inds_org = np.arange(self.natoms, dtype=int)
                goods = np.delete(inds_org, inds)
                self.atoms = self.atoms.iloc[goods]
                self.initialization()
        return inds, org_xyzns, thisxyznsq, thistypes

    def select_by_coords_wrt_center(self, distances, center=[0.5, 0.5, 0.5], is_cartesian=False, depress=1,
                         style=0, delete=False, delete_style="OUTSIDE", sort=False):

        if isinstance(distances, int) or isinstance(distances, float):
            distances = [distances] * 3
        distances = np.array(distances)
        fdistances= np.dot(distances, self.box.inv_matrix)

        self.insert_itags()
        center = np.array(center)
        thistypes = self.atoms['type'].to_numpy()
        inf = float("inf")
        xyzlims = [[-inf, inf], [-inf, inf], [-inf, inf]]
        if style == 0:
            Fractional = True
        else:
            Fractional = False
        if style == 0:
            if is_cartesian:
                center = np.dot(center, self.box.inv_matrix)
            org_xyzns = np.vstack((self.atoms["xsn"], self.atoms["ysn"], self.atoms["zsn"]))
            thisxyzns = copy.deepcopy(org_xyzns)
            for i in range(3):
                d = fdistances[i] * 0.5
                if isinstance(depress, int) and i == depress:
                    thisxyzns[depress] = np.zeros(len(thisxyzns[depress]))
                    center[depress] = 0.0
                else:
                    xyzlims[i][0] = center[i] - d
                    xyzlims[i][1] = center[i] + d

            xyzns = thisxyzns.T - center
            xyzns = xyzns.T
            xyzns[0] = lmpData.normalize_frac_coords(xyzns[0], style=0)
            xyzns[1] = lmpData.normalize_frac_coords(xyzns[1], style=0)
            xyzns[2] = lmpData.normalize_frac_coords(xyzns[2], style=0)
            thisxyznsq = np.sum(np.dot(xyzns.T, self.box.matrix) ** 2, axis=1)
        else:
            org_xyzns = np.vstack((self.atoms["x"], self.atoms["y"], self.atoms["z"]))
            thisxyzns = copy.deepcopy(org_xyzns)
            if not is_cartesian:
                center = np.dot(center, self.box.matrix)
            for i in range(3):
                d = distances[i] * 0.5
                if isinstance(depress, int) and i == depress:
                    thisxyzns[depress] = np.zeros(len(thisxyzns[depress]))
                    center[depress] = 0.0
                else:
                    xyzlims[i][0] = center[i] - d
                    xyzlims[i][1] = center[i] + d

            xyzns = thisxyzns.T - center
            thisxyznsq = np.sum(xyzns ** 2, axis=1)

        inds = self.select_by_coords(xlim=xyzlims[0], ylim=xyzlims[1], zlim=xyzlims[2],
                         Fractional=Fractional, style="INCLUDE", delete=False)
        org_xyzns = (org_xyzns.T)[inds]
        if style == 0:
            org_xyzns = np.dot(org_xyzns, self.box.matrix)
        thisxyznsq = thisxyznsq[inds]
        thisxyznsq = np.sqrt(thisxyznsq)
        thistypes = thistypes[inds]

        if sort:
            isorts = np.argsort(thisxyznsq)
            inds = inds[isorts]
            org_xyzns = org_xyzns[isorts]
            thisxyznsq = thisxyznsq[isorts]
            thistypes = thistypes[isorts]

        if delete:
            if delete_style[0:3].upper() == "OUT":
                self.atoms = self.atoms.iloc[inds]
                self.initialization()
            else:
                inds_org = np.arange(self.natoms, dtype=int)
                goods = np.delete(inds_org, inds)
                self.atoms = self.atoms.iloc[goods]
                self.initialization()
        return inds, org_xyzns, thisxyznsq, thistypes

    def add_an_entry(self, indict, loc=None, ff_elements=myElements, atomic_masses=None, check_distance=False,
                     rcut=1.0):
        if check_distance:
            orgdata = self.deepcopy()
        default_dict = self.generate_default_dict()
        for key in indict:
            if key in default_dict:
                default_dict[key] = indict[key]

        if isinstance(loc, int):
            self.atoms.loc[loc] = default_dict
        else:
            self.atoms.loc[self.idmax + 1] = default_dict
            self.idmax += 1

        if default_dict['type'] > self.ntypes:
            self.assert_force_field(ff_elements, atomic_masses=atomic_masses)

        self.initialization()
        if check_distance:
            coords = np.array([default_dict['x'], default_dict['y'], default_dict['z']])
            inds, xyzs, distances, types = lmpData.compute_site_distance(coords, orgdata,
                                                                         style=0, rcut=rcut, sort=False)
            if len(inds) > 0:
                print(f"following atoms have a distance shorter than {rcut}")
                print(f"itags:{inds} distance: {distances}")
                print(f"coords: {xyzs}")
                print("--------")

    def replace_an_entry(self, i, indict, style="itag", ff_elements=myElements, atomic_masses=None):
        if style == "itag":
            orgdict = self.atoms.iloc[i].to_dict(index=True)
        else:
            orgdict = self.atoms.loc[i].to_dict(index=True)
        loc = orgdict["index"]
        del orgdict["index"]
        for key in indict:
            if key in orgdict:
                orgdict[key] = indict[key]

        if orgdict['type'] > self.ntypes:
            self.assert_force_field(ff_elements, atomic_masses=atomic_masses)

        self.atoms.insert(orgdict, loc)
        self.initialization()

    def sort(self, by='index', Ascending=True):
        if by.upper() == "INDEX":
            self.atoms.sort_index(ascending=Ascending, inplace=True)
        else:
            self.atoms.sort_values(by=by, ascending=Ascending, inplace=True)

    def zero_coords(self, thres=[0.1, 0.1, 0.1]):
        thres = np.array(thres)
        fthres = np.dot(thres, self.box.inv_matrix)
        cols = ["xsn", "ysn", "zsn"]
        for i in range(len(cols)):
            f = self.atoms[cols[i]]
            f = lmpData.normalize_frac_coords(f, style=1)
            f = np.select([f < 1.0 - fthres[i], f >= 1.0 - fthres[i]], [f, f - 1.0])
            if cols[i] in self.atoms.columns: self.atoms = self.atoms.drop(cols[i], axis=1)
            self.atoms.insert(len(self.atoms.columns), cols[i], f)
        self.fracts2coords()

    def set_coord_mins_to_zeros(self, directions=[0, 1, 2]):
        if isinstance(directions, int):
            directions = [directions]
        for i in directions:
            if i == 0:
                v = self.atoms['x'].to_numpy()
                vmin = np.min(v)
                self.atoms['x'] = v - vmin
            elif i == 1:
                v = self.atoms['y'].to_numpy()
                vmin = np.min(v)
                self.atoms['y'] = v - vmin
            elif i == 2:
                v = self.atoms['z'].to_numpy()
                vmin = np.min(v)
                self.atoms['z'] = v - vmin
        self.coords2fracts()

    def find_center(self, zero_coords=False, thres=[0.1, 0.1, 0.1]):
        if zero_coords:
            self.zero_coords(thres=thres)
        frac_center = [np.mean(self.atoms["xsn"]), np.mean(self.atoms["ysn"]), np.mean(self.atoms["zsn"])]
        center = [np.mean(self.atoms["x"]), np.mean(self.atoms["y"]), np.mean(self.atoms["z"])]
        return np.array(frac_center), np.array(center)

    def find_center_atom_coords(self, r, center=[0.5, 0.5, 0.5], is_cartesian=False, style=1):
        inds, xyzs, ds, types = self.select_by_radius(r, center=center, is_cartesian=is_cartesian, depress=None,
                                                      style=style, delete=False, sort=True)
        return inds[0], xyzs[0], ds[0], types[0]

    def center_atoms(self, thres=[0.1, 0.1, 0.1]):
        frac_cen, cen = self.find_center(zero_coords=True, thres=thres)
        self.atoms['xsn'] += 0.5 - frac_cen[0]
        self.atoms['ysn'] += 0.5 - frac_cen[1]
        self.atoms['zsn'] += 0.5 - frac_cen[2]
        self.fracts2coords()

    @staticmethod
    def modify_by_symmetry(df, symmop, normalization=True):
        cols = ["x", "y", "z"]
        xyzs = np.vstack((df[cols[0]], df[cols[1]], df[cols[2]]))
        orgmin = np.array([np.min(xyzs[0]), np.min(xyzs[1]), np.min(xyzs[2])])
        xyzs = symmop.operate_multi(xyzs.T)
        xyzs = xyzs.T
        if normalization:
            newmin = np.array([np.min(xyzs[0]), np.min(xyzs[1]), np.min(xyzs[2])])
            origin = orgmin - newmin
        else:
            origin = np.zeros(3)
        df["x"] = xyzs[0] + origin[0]
        df["y"] = xyzs[1] + origin[1]
        df["z"] = xyzs[2] + origin[2]
        return df

    def modify_atoms(self, translation=None, rotation=None, inds=None, is_cartesian=True,
                     center_coords=False, normalization4symmop=False, normalization=True, types=None,
                     Ang_Format="DEGREE", Ang_Style="EU", ff_elements=myElements, atomic_masses=None,
                     thres=[0.1, 0.1, 0.1]):

        ind_org = np.arange(self.natoms, dtype=int)
        ind_del = np.array([], dtype=int)
        if inds is None:
            inds = np.arange(self.natoms, dtype=int)
        elif isinstance(inds, int) or isinstance(inds, np.int_):
            inds = np.array([inds])
        else:
            inds = np.array(inds)

        thisatoms = self.atoms.copy(deep=True)
        thisatoms = thisatoms.iloc[inds]

        if len(inds) < self.natoms:
            ind_del = np.delete(ind_org, inds)
            del_atoms = self.atoms.copy(deep=True)
            del_atoms = del_atoms.iloc[ind_del]

        if rotation is None:
            rot_matrix = np.eye(3, dtype=float)
        else:
            rot_matrix = np.array(rotation)
            if rot_matrix.shape == (3, 3):
                pass
            elif rot_matrix.shape == (3,):
                rot_matrix = generate_rotation_matrix(rot_matrix, Ang_Format=Ang_Format, Ang_Style=Ang_Style)
                rot_matrix = rot_matrix.T

        if translation is not None:
            origin = np.array(translation)
            if not is_cartesian:
                origin = np.dot(origin, self.box.matrix)
        else:
            origin = np.zeros(3, dtype=float)

        symmop = SymmOp.from_rotation_and_translation(rot_matrix, origin)
        thisatoms = lmpData.modify_by_symmetry(thisatoms, symmop, normalization=normalization4symmop)

        reinit = False
        if isinstance(types, int):
            thisatoms['type'] = [types] * len(thisatoms)
            if types > self.ntypes:
                self.assert_force_field(ff_elements, atomic_masses=atomic_masses)
            reinit = True
        elif isinstance(types, list) or isinstance(types, np.ndarray):
            thisatoms['type'] = types
            if np.max(np.array(types)) > self.ntypes:
                self.assert_force_field(ff_elements, atomic_masses=atomic_masses)
            reinit = True

        if len(inds) < self.natoms:
            self.atoms = pd.concat([thisatoms, del_atoms])
        else:
            self.atoms = thisatoms.copy(deep=True)

        if center_coords:
            self.center_atoms(thres=thres)

        if reinit:
            self.initialization(normalization=normalization, style=1)
        else:
            self.coords2fracts(normalization=normalization, style=1)
        if normalization: self.fracts2coords()

    def make_supercell(self, supercell, ff_elements=None, is_sort=False):
        s = self.to_structure()
        s.make_supercell(supercell)
        newdata = lmpData.from_pmg_structure(s, self.atom_style, ff_elements=ff_elements, is_sort=is_sort)
        newdata.initialization()
        return newdata

    def make_supercell_simple(self, supercell):
        supercell = np.array(supercell)
        if len(supercell.shape) > 1:
            raise ValueError("It can only do [int1, int2, int3] type of supercell")

        oldmatrix = copy.deepcopy(self.box.matrix)
        newmatrix = np.zeros((3, 3), dtype=float)
        newmatrix[0] = oldmatrix[0] * supercell[0]
        newmatrix[1] = oldmatrix[1] * supercell[1]
        newmatrix[2] = oldmatrix[2] * supercell[2]
        lengths = copy.deepcopy(self.box.lengths)

        xsn = self.atoms['xsn'].to_numpy()
        ysn = self.atoms['ysn'].to_numpy()
        zsn = self.atoms['zsn'].to_numpy()
        for i in range(supercell[0]):
            thisxsn = (xsn + i) / supercell[0]
            x = self.atoms['x'].to_numpy() + i * lengths[0]
            for j in range(supercell[1]):
                thisysn = (ysn + j) / supercell[1]
                y = self.atoms['y'].to_numpy() + j * lengths[1]
                for k in range(supercell[2]):
                    thiszsn = (zsn + k) / supercell[2]
                    z = self.atoms['z'].to_numpy() + k * lengths[2]
                    if i == 0 and j == 0 and k == 0:
                        totatoms = self.atoms.copy(deep=True)
                        totatoms['x'] = x
                        totatoms['y'] = y
                        totatoms['z'] = z
                        totatoms['xsn'] = thisxsn
                        totatoms['ysn'] = thisysn
                        totatoms['zsn'] = thiszsn
                    else:
                        thisatoms = self.atoms.copy(deep=True)
                        thisatoms['x'] = x
                        thisatoms['y'] = y
                        thisatoms['z'] = z
                        thisatoms['xsn'] = thisxsn
                        thisatoms['ysn'] = thisysn
                        thisatoms['zsn'] = thiszsn
                        totatoms = pd.concat([totatoms, thisatoms])
        totatoms = totatoms.reset_index(drop=True)
        totatoms = totatoms.set_index(np.arange(len(totatoms), dtype=int) + 1)
        box, symmop = lattice_2_lmpbox(Lattice(newmatrix))
        newdata = lmpData(box, self.masses, totatoms, atom_style=self.atom_style, force_field=self.force_field)
        return newdata

    def swap_axes(self, newaxis):
        xyzs = np.vstack((self.atoms['x'], self.atoms['y'], self.atoms['z']))
        self.atoms['x'] = xyzs[newaxis[0]]
        self.atoms['y'] = xyzs[newaxis[1]]
        self.atoms['z'] = xyzs[newaxis[2]]
        oldm = copy.deepcopy(self.box.matrix)
        newmatrix = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                newmatrix[i][j] = oldm[newaxis[i]][newaxis[j]]
        newlatt = Lattice(newmatrix)
        self.box, symmop = lattice_2_lmpbox(newlatt)
        self.atoms = lmpData.modify_by_symmetry(self.atoms, symmop)
        self.initialization()

    def modify_lmpbox(self, newmatrix, style=0, reset_ids=True):
        #style = 0: keep fractional coords
        #style = 1: keep cartesian coords
        #style = 2: remove fraction >= 1.0 or fraction < 0
        newlatt = Lattice(newmatrix)
        if style == 0:
            self.box, symmop = lattice_2_lmpbox(newlatt)
            self.fracts2coords()
        elif style == 1:
            self.box, symmop = lattice_2_lmpbox(newlatt)
            #symmop = find_symmop_lattices(newlatt, self.box.to_lattice())
            #self.atoms = lmpData.modify_by_symmetry(self.atoms, symmop, normalization=True)
            self.coords2fracts(normalization=False, axes=[0, 1, 2])
        elif style == 2:
            self.modify_atoms(translation=[0.001, 0.001, 0.001], rotation=None, is_cartesian=True, inds=None)
            self.box, symmop = lattice_2_lmpbox(newlatt)
            #symmop = find_symmop_lattices(newlatt, self.box.to_lattice())
            #self.atoms = lmpData.modify_by_symmetry(self.atoms, symmop, normalization=True)
            self.coords2fracts(normalization=False, axes=[0, 1, 2])

            bads = []
            for i in range(len(self.atoms)):
                d = self.atoms.iloc[i].to_dict()
                fxyz = np.array([d["xsn"], d["ysn"], d["zsn"]])
                isvalid = True
                for idim in range(3):
                    if fxyz[idim] < 0 or fxyz[idim] > 1.0:
                        isvalid = False
                if not isvalid:
                    bads.append(i)

            self.select_by_coords(xlim=[0.0, 1.0],
                                  ylim=[0.0, 1.0],
                                  zlim=[0.0, 1.0],
                                  Fractional=True, style="INCLUSION", delete=True)
            self.fracts2coords()
            self.modify_atoms(translation=[-0.001, -0.001, -0.001], rotation=None, is_cartesian=True, inds=None)
            if reset_ids:
                self.reset_atom_ids()

    def distort_data(self, distort, distorttype, dca=0, ca0=1.60, crystal="bcc"):
        if distorttype == "EOS":
            da = Distortion.EOS_dis(distort)
        elif distorttype == "EOS4HCP":
            da = Distortion.EOS_dis_hcp(distort, dca, ca0)
        elif distorttype == "TD":
            da = Distortion.tetr_dis(distort)
        elif distorttype == "OD":
            da = Distortion.orth_dis(distort)
        elif distorttype == "MD":
            da = Distortion.mono_dis(distort, crystal=crystal)
        else:
            da = np.eye(3)
            print("Uncoded distorttype!")

        thisbox = Distortion.apply_distortion(da, self.box.matrix)
        newlatt = Lattice(thisbox)
        self.box, symmop = lattice_2_lmpbox(newlatt)
        self.fracts2coords()

    def scale_data(self, app_scale, style=0):
        """
        :param app_scale:
        :param style: {1: scale_catesian and keep box unchanged, else: scale both
        :return:
        """
        orgbox = copy.deepcopy(self.box.matrix)
        s0 = 1.0
        s0 *= app_scale
        newlatt = Lattice(s0 * orgbox)
        self.box, symmop = lattice_2_lmpbox(newlatt)
        if style == 1:
            self.fracts2coords()
            orglatt = Lattice(orgbox)
            self.box, symmop = lattice_2_lmpbox(orglatt)
            self.coords2fracts(normalization=False)
        else:
            self.fracts2coords()

    def add_vacuum(self, lvac=20.0, direction=2, zero_coords=True, thres=[0.1, 0.1, 0.1]):
        if zero_coords:
            self.zero_coords(thres=thres)
        newmatrix = copy.deepcopy(self.box.matrix)
        scale = (self.box.lengths[direction] + lvac) / self.box.lengths[direction]
        newmatrix[direction, :] *= scale
        newlatt = Lattice(newmatrix)
        self.box, symmop = lattice_2_lmpbox(newlatt)
        cols = ["x", "y", "z"]
        self.atoms[cols[direction]] += lvac / 2.0
        self.coords2fracts(normalization=False)

    def assert_atom_types(self, iatoms, randomize=False):
        iatoms = np.array(iatoms)
        if np.sum(iatoms) != self.natoms:
            raise ValueError("Summation of iatoms must be equal to number of atoms!")
        ntypes = len(iatoms)
        types = []
        for i in range(ntypes):
            types += [i + 1] * iatoms[i]
        types = np.array(types)
        if randomize:
            np.random.shuffle(types)
        self.atoms['type'] = types
        self.get_data_info()

    def get_interface_scan(self, nx, ny, xrange, yrange, thres=[0.1, 0.1, 0.1]):
        frac_center, center = self.find_center(zero_coords=True, thres=thres)
        list_data = []
        list_inddict = []
        for ix in range(nx):
            for iy in range(ny):
                t = self.deepcopy()
                a = ix * xrange / max((nx - 1), 1)
                b = iy * yrange / max((ny - 1), 1)
                c = 0.0
                shifts = np.array([a, b, c])
                shifts = np.dot(shifts, t.box.matrix)
                t.atoms['x'] = np.select([t.atoms['z'] < center[2] - 0.1, t.atoms['z'] >= center[2] - 0.1],
                                         [t.atoms['x'], t.atoms['x'] + shifts[0]])
                t.atoms['y'] = np.select([t.atoms['z'] < center[2] - 0.1, t.atoms['z'] >= center[2] - 0.1],
                                         [t.atoms['y'], t.atoms['y'] + shifts[1]])
                t.coords2fracts(normalization=True)
                list_data.append(t)
                inddict = {"ix": ix, "iy": iy}
                list_inddict.append(inddict)
        return list_data, list_inddict

    def get_molecule_scan(self, scales):
        list_data = []
        for i in range(len(scales)):
            t = self.deepcopy()
            app_scale = scales[i]
            t.scale_data(app_scale, style=1)
            list_data.append(t)
        return list_data

    def get_distortions(self, distortions, distorttype, dca=0, ca0=1.60, crystal="bcc"):
        list_data = []
        for i in range(len(distortions)):
            t = self.deepcopy()
            app_strain = distortions[i]
            t.distort_data(app_strain, distorttype, dca=dca, ca0=ca0, crystal=crystal)
            list_data.append(t)
        return list_data

    def get_AIMD(self, symbols, iatoms, temp):
        thermal_exp = lmpData.find_thermal_expansion(symbols, iatoms, temp)
        self.scale_data(1.0 + thermal_exp, style=0)

    def create_edge_dislocation(self, burgerm, edge_style=0, nedges=1, fy_start=None, fz_start=0.5,
                                add_vacuum=False, direction=2, lvac=20.0,
                                reset_ids=True, thres=[0.1, 0.1, 0.1]):
        self.zero_coords(thres=thres)
        buffer = burgerm * 0.03
        fthres = np.dot(np.array(thres), self.box.inv_matrix)
        fyshift_lower = [0.0, burgerm / 2.0 + buffer, 0.0]
        fyshift_lower = np.dot(np.array(fyshift_lower), self.box.inv_matrix)
        yshift_upper = [0.0, burgerm / 2.0 - buffer, 0.0]
        fyshift_upper = np.dot(np.array(yshift_upper), self.box.inv_matrix)
        if nedges == 1:
            if isinstance(fy_start, float):
                fy_start = [fy_start]
            else:
                fy_start = [0.5]
            if fz_start == 0.0:
                zmin = [0.0 - fthres[2]]
                zmax = [0.5 - fthres[2]]
            else:
                zmin = [0.5 - fthres[2]]
                zmax = [1.0]
        else:
            if isinstance(fy_start, float):
                fy_start = [fy_start, fy_start + 0.5]
            else:
                fy_start = [0.25, 0.75]
            zmin = [0.5 - fthres[2], 0 - fthres[2]]
            zmax = [1.0, 0.5 - fthres[2]]
        for i in range(nedges):
            xlim = [-fthres[0], 1.0]
            ylim = [fy_start[i] - fyshift_lower[1], fy_start[i] + fyshift_upper[1]]
            zlim = [zmin[i], zmax[i]]
            selected_inds = self.select_by_coords(xlim=xlim, ylim=ylim, zlim=zlim,
                                  Fractional=True, style="INCLUDE", delete=False)
            if edge_style == 0:
                self.remove_by_inds(selected_inds, style="dyn")
            else:
                if len(selected_inds) > 0:
                    selected_inds = np.array(selected_inds)
                    extra_plane = self.atoms.iloc[selected_inds].copy(deep=True)
                    extra_plane['x'] += burgerm / 4.0
                    extra_plane['y'] += burgerm / 2.0
                    self.atoms = pd.concat([self.atoms, extra_plane])
        self.initialization(normalization=True, style=1)
        newmatrix = copy.deepcopy(self.box.matrix)
        if edge_style == 0:
            scale = (self.box.lengths[1] - 0.5 * nedges * burgerm) / self.box.lengths[1]
        else:
            scale = (self.box.lengths[1] + 0.5 * nedges * burgerm) / self.box.lengths[1]
        newmatrix[1, :] *= scale
        self.modify_lmpbox(newmatrix, style=0)
        if add_vacuum:
            thres = [0.1, 0.1, 0.1]
            thres[direction] = burgerm / 2 + buffer
            self.add_vacuum(lvac=lvac, direction=direction, zero_coords=True, thres=thres)
        if reset_ids:
            self.reset_atom_ids()

    def create_screw_dislocation(self, burgerm, nscrews, style="bcc", handle_pbc="tilt", orientation=True,
                                 add_vacuum=False, direction=0, lvac=20.0, thres=[0.1, 0.1, 0.1]):
        """
        for bcc
        a = [1, -1, 0]
        b = [1, 1, -2]
        c = [1, 1, 1]/2
        for fcc
        a = [1, 1, 1]
        b = [1, 1, -2]/2
        c = [1, -1, 0]/2
        """
        buffer = burgerm * 0.03
        if not isinstance(direction, int):
            direction = 0
        self.zero_coords(thres=thres)
        glide_types_bcc = [1, 0, 3, 2]
        glide_types_fcc = [0, 1, 2, 3]
        theta = - np.pi / 3
        rotmat4hcp = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]], dtype=float)
        if style == "bcc":
            lattpara = 2.0 * burgerm / np.sqrt(3)
            lenatom = lattpara * np.sqrt(6) / 6.0
            nlayers_del = 2.0
            glide_types = copy.deepcopy(glide_types_bcc)
        elif style == "fcc":
            lattpara = 2.0 * burgerm / np.sqrt(2)
            lenatom = lattpara * np.sqrt(6) / 2.0 / 6.0
            nlayers_del = 3.0
            glide_types = copy.deepcopy(glide_types_fcc)
        else:
            raise ValueError("Only styles of bcc and fcc are coded!")

        if nscrews == 1 and handle_pbc.upper() == "TILT":
            if add_vacuum:
                print("Warning: It is not a valid setting to make one screw dislocations.")
                print("See Examples/INPUTs/Dislocations/make_dislocations.py to make one dislocation")

        ifchopatoms = False
        iftilt = False
        if nscrews == 1:
            burgerms = [burgerm]
            applied_directions = [True]
            glides = [glide_types[0]]
            start_positions = []
            coords = np.dot(np.array([0.5, 0.5, 0.5]), self.box.matrix)
            ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                 is_cartesian=True, style=1)
            if self.box.is_orthogonal:
                if handle_pbc == "tilt":
                    iftilt = True
                else:
                    ifchopatoms = True

            if ifchopatoms:
                coords[1] -= burgerm * np.sqrt(2) / 3.0
                coords[2] -= burgerm / 6.0
            fxyz = np.dot(self.box.inv_matrix.T, coords)
            start_positions.append(fxyz)

        elif nscrews == 2:
            burgerms = [burgerm, -burgerm]
            applied_directions = [True, False]
            #glides = [glide_types[0], glide_types[2]] #I think it also works
            glides = [glide_types[0], glide_types[0]]  #I think it also works
            start_positions = []
            if self.box.is_orthogonal:
                coords = np.dot(np.array([0.5, 0.25, 0.5]), self.box.matrix)
                ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                     is_cartesian=True, style=1)
                fxyz = np.dot(self.box.inv_matrix.T, coords)
                start_positions.append(fxyz)
                coords = np.dot(np.array([0.5, 0.75, 0.5]), self.box.matrix)
                ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                     is_cartesian=True, style=1)
                fxyz = np.dot(self.box.inv_matrix.T, coords)
                start_positions.append(fxyz)
            else:
                coords = np.dot(np.array([1 / 3, 2 / 3, 0.5]), self.box.matrix)
                ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                     is_cartesian=True, style=1)
                fxyz = np.dot(self.box.inv_matrix.T, coords)
                start_positions.append(fxyz)
                coords = np.dot(np.array([2 / 3, 1 / 3, 0.5]), self.box.matrix)
                ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                     is_cartesian=True, style=1)
                fxyz = np.dot(self.box.inv_matrix.T, coords)
                start_positions.append(fxyz)
        elif nscrews == 4:
            burgerms = [burgerm, -burgerm, -burgerm, burgerm]
            applied_directions = [True, False, False, True]
            glides = [glide_types[0], glide_types[0], glide_types[0], glide_types[0]]
            start_positions = []
            coords = np.dot(np.array([0.25, 0.25, 0.5]), self.box.matrix)
            ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                 is_cartesian=True, style=1)
            fxyz = np.dot(self.box.inv_matrix.T, coords)
            start_positions.append(fxyz)
            coords = np.dot(np.array([0.25, 0.75, 0.5]), self.box.matrix)
            ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                 is_cartesian=True, style=1)
            fxyz = np.dot(self.box.inv_matrix.T, coords)
            start_positions.append(fxyz)
            coords = np.dot(np.array([0.75, 0.25, 0.5]), self.box.matrix)
            ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                 is_cartesian=True, style=1)
            fxyz = np.dot(self.box.inv_matrix.T, coords)
            start_positions.append(fxyz)
            coords = np.dot(np.array([0.75, 0.75, 0.5]), self.box.matrix)
            ind, coords, d, itype = self.find_center_atom_coords(burgerm + buffer, center=coords,
                                                                 is_cartesian=True, style=1)
            fxyz = np.dot(self.box.inv_matrix.T, coords)
            start_positions.append(fxyz)
        else:
            raise ValueError("Uncoded number of screw dislocations.")

        for iscrew in range(nscrews):
            thisb = burgerms[iscrew]
            start_position = copy.deepcopy(start_positions[iscrew][:])
            applied_direction = applied_directions[iscrew]
            glide = glides[iscrew]

            if applied_direction:
                x = self.atoms['xsn'] - start_position[0]
                y = self.atoms['ysn'] - start_position[1]
            else:
                x = -(self.atoms['xsn'] - start_position[0])
                y = -(self.atoms['ysn'] - start_position[1])
            z = np.zeros(len(x))
            fxyzs = np.vstack((x, y, z))
            xyzs = np.dot(self.box.matrix.T, fxyzs)
            xys = np.vstack((xyzs[0], xyzs[1]))
            if not self.box.is_orthogonal:
                if glide >= 2:
                    xys = np.dot(rotmat4hcp, xys)
            if glide % 2 == 0:
                angles = np.arctan2(xys[1], xys[0])
            elif glide % 2 == 1:
                angles = np.arctan2(xys[0], xys[1])
            disps = -0.5 * thisb * angles / pi
            self.atoms['z'] += disps
            self.coords2fracts(normalization=True)

        if ifchopatoms:
            fyshift = [0.0, (nlayers_del + 0.5) * lenatom, 0.0]
            fyshift = np.dot(fyshift, self.box.inv_matrix)

            vmin = np.dot(np.array([-0.1, -0.1, -0.1]), self.box.inv_matrix)
            vmax = [1.0, 1.0 - fyshift[1], 1.0]
            self.select_by_coords(xlim=[vmin[0], vmax[0]], ylim=[vmin[1], vmax[1]],
                                  zlim=[vmin[2], vmax[2]],
                                  Fractional=True, style="INCLUDE", delete=True)

            newmatrix = copy.deepcopy(self.box.matrix)
            scale = (self.box.lengths[1] - nlayers_del * lenatom) / self.box.lengths[1]
            newmatrix[1, :] *= scale
            self.modify_lmpbox(newmatrix, style=1)

        if iftilt:
            newmatrix = copy.deepcopy(self.box.matrix)
            if direction == 1:
                tdirection = 0
            else:
                tdirection = 1
            newmatrix[2, tdirection] = burgerm / 2
            self.modify_lmpbox(newmatrix, style=1)

        if add_vacuum:
            thres = [0.1, 0.1, 0.1]
            thres[direction] = burgerm / 2 + buffer
            self.add_vacuum(lvac=lvac, direction=direction, zero_coords=True, thres=thres)

        if orientation:
            newaxis = [2, 1, 0]
            self.swap_axes(newaxis)

    def linear_interpolate(self, findata, nimages, thres=[0.1, 0.1, 0.1]):
        thres = np.array(thres)
        self.zero_coords(thres=thres)
        findata.zero_coords(thres=thres)

        list_data = []
        for im in range(0, nimages + 2):
            b = im / (nimages + 1)
            a = (nimages + 1 - im) / (nimages + 1)
            t = self.deepcopy()
            t.atoms['x'] = a * self.atoms['x'] + b * findata.atoms['x']
            t.atoms['y'] = a * self.atoms['y'] + b * findata.atoms['y']
            t.atoms['z'] = a * self.atoms['z'] + b * findata.atoms['z']
            t.initialization()
            list_data.append(t)
        return list_data

    def append_data(self, indata, direction=2, box_handle="unchanged", add_vacuum=False, lvac=20.0,
                    typesoffset=False, ff_elements=["Fe", "Fe"], atomic_masses=[55.845, 55.845],
                    reset_ids=True, thres=[0.1, 0.1, 0.1]):
        """
        box_handle means how to deal with the rest two lattice vectors: "mean" is average. Otherwise, uses self
        """
        outdata = self.deepcopy()
        if typesoffset:
            print("The C15 have offset type by 1.")
            print("you must input ff_elements and atomic_masses accoordingly.")
            indata.atoms['type'] += 1
            outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)
        newmatrix = copy.deepcopy(outdata.box.matrix)
        if box_handle == "mean":
            vsame = copy.deepcopy(newmatrix[direction])
            newmatrix += indata.box.matrix
            newmatrix[direction] = vsame
            self.modify_lmpbox(newmatrix, style=0)

            matrix2 = copy.deepcopy(newmatrix)
            matrix2[direction] = indata.box.matrix[direction]
            indata.modify_lmpbox(matrix2, style=0)

        scale = (outdata.box.lengths[direction] + indata.box.lengths[direction]) / outdata.box.lengths[direction]
        newmatrix[direction, :] *= scale
        newbox, symmop = lattice_2_lmpbox(Lattice(newmatrix))
        scale1 = outdata.box.lengths[direction] / newbox.lengths[direction]
        scale2 = indata.box.lengths[direction] / newbox.lengths[direction]
        cols = ['xsn', 'ysn', 'zsn']
        indata.atoms[cols[direction]] = scale1 + indata.atoms[cols[direction]] * scale2
        indata.fracts2coords()
        outdata.atoms = pd.concat([outdata.atoms, indata.atoms])
        outdata.box = newbox
        outdata.initialization()
        if add_vacuum:
            outdata.add_vacuum(lvac=lvac, direction=direction, zero_coords=True, thres=thres)
        if reset_ids:
            outdata.reset_atom_ids()
        return outdata

    def mergy_data(self, indata,  translation=None, rotation=None, depress=1, is_cartesian=True,
                   normalization4symmop=False, normalization=True, check_distance=False, rcut=1.0,
                   modify_box=False, newmatrix=None, style=0,
                   add_vacuum=False, lvac=20.0, direction=2,
                   typesoffset=False, ff_elements=["Fe", "Fe"], atomic_masses=[55.845, 55.845],
                   reset_ids=True, thres=[0.1, 0.1, 0.1]):
        outdata = self.deepcopy()

        if typesoffset:
            print("The C15 have offset type by 1.")
            print("you must input ff_elements and atomic_masses accoordingly.")
            indata.atoms['type'] += 1
            outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)

        if translation is not None:
            translation = np.array(translation)
            if not is_cartesian:
                translation = np.dot(translation, outdata.box.matrix)
                is_cartesian = True
            if isinstance(depress, int):
                translation[depress] += 0.5 * outdata.box.lengths[depress]

        indata.modify_atoms(translation=translation, rotation=rotation, is_cartesian=is_cartesian,
                            center_coords=False, normalization4symmop=normalization4symmop, normalization=False)

        outdata.atoms = pd.concat([outdata.atoms, indata.atoms])
        outdata.initialization(normalization=normalization, style=1)

        if modify_box:
            outdata.modify_lmpbox(newmatrix, style=style)

        if check_distance:
            for i in range(indata.natoms):
                d = indata.atoms.iloc[i].to_dict()
                coords = np.array([d["x"], d["y"], d["z"]])
                inds, xyzs, distances, types = lmpData.compute_site_distance(coords, self,
                                                                             rcut=rcut, style=0, sort=False)
                if len(inds) > 0:
                    print(f"itag in tmpdat: {i} coords: {coords}")
                    print(f"following atoms have a distance shorter than {rcut}")
                    print(f"itags:{inds} distance: {distances}")
                    print(f"coords: {xyzs}")
                    print("--------")

        if add_vacuum:
            outdata.add_vacuum(lvac=lvac, direction=direction, zero_coords=True, thres=thres)

        if reset_ids:
            outdata.reset_atom_ids()
        return outdata

    def merge_data_with_splits(self, indata, bondlength,
                               to_center=[0.5, 0.5, 0.5], is_cartesian=False,
                               splits=[-0.25, -0.25, -0.25], rcut=3.0, tolerance=3.0,
                               mergy_style=0,
                               modify_box=False, newmatrix=None, style=0,
                               add_vacuum=False, lvac=20.0, direction=2,
                               typesoffset=False, ff_elements=["Fe", "Fe"], atomic_masses=[55.845, 55.845],
                               reset_ids=True, thres=[0.1, 0.1, 0.1]):
        """
        :param indata: data to be inserted
        :param bondlength: bondlength
        :param to_center: the center of indata will move to to_center of substrate (self)
        :param is_cartesian: is to_center cartesian
        :param splits: ratio to bondlength, splits on data to be inserted, on substrate will opposite sign
        :param rcut: rcut to find atoms in substrate of a given atom in indata
        :param tolerance: judgement of atoms overlapping. these two atoms will be splitted
        :param modify_box: will modify box
        :param newmatrix: new box matrix
        :param style: style of modify box
        :param add_vacuum: will add vacuum
        :param lvac: length of vac
        :param direction: direction of adding vacuum
        :param reset_ids: reset_ids?
        :return: a new lmpData
        """
        outdata = self.deepcopy()
        buffer = bondlength * 0.03
        splits = np.array(splits)
        if typesoffset:
            print("The indata has offset type by 1.")
            print("you must input ff_elements and atomic_masses accoordingly.")
            indata.atoms['type'] += 1
            outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)

        ind, subcen, d, itype = outdata.find_center_atom_coords(bondlength + buffer,
                                                                center=to_center, is_cartesian=is_cartesian, style=1)

        xyzs = np.vstack((indata.atoms["x"], indata.atoms["y"], indata.atoms["z"]))
        coords = [np.mean(xyzs[0]), np.mean(xyzs[1]), np.mean(xyzs[2])]
        ind, laycen, d, itype = indata.find_center_atom_coords(bondlength + buffer,
                                                               center=coords, is_cartesian=True, style=1)

        translation = subcen - laycen
        selected_inds = []
        for i in range(indata.natoms):
            d = indata.atoms.iloc[i].to_dict()
            coords = np.array([d["x"], d["y"], d["z"]])
            coords += translation
            inds, xyzs, ds, types = lmpData.compute_site_distance(coords, self,
                                                                  rcut=rcut, style=0, sort=True)
            if len(ds) > 0:
                if ds[0] < tolerance:
                    thissplits = splits * (bondlength - ds[0])
                    for idim in range(3):
                        diff = xyzs[0][idim] - coords[idim]
                        if diff >= -tolerance/5:
                            if thissplits[idim] >= 0.0:
                                coords[idim] -= thissplits[idim]
                                xyzs[0][idim] += thissplits[idim]
                            else:
                                coords[idim] += thissplits[idim]
                                xyzs[0][idim] -= thissplits[idim]
                        else:
                            if thissplits[idim] >= 0.0:
                                coords[idim] += thissplits[idim]
                                xyzs[0][idim] -= thissplits[idim]
                            else:
                                coords[idim] -= thissplits[idim]
                                xyzs[0][idim] += thissplits[idim]
                    #print(f"ds0:{ds[0]} after:{np.linalg.norm(xyzs[0] - coords)}")
                    ind = inds[0]
                    '''
                    subd = outdata.atoms.iloc[ind].to_dict()
                    d["x"] = coords[0]
                    d["y"] = coords[1]
                    d["z"] = coords[2]
                    subd["x"] = xyzs[0][0]
                    subd["y"] = xyzs[0][1]
                    subd["z"] = xyzs[0][2]
                    indata.atoms.iloc[i] = d
                    outdata.atoms.iloc[ind] = subd
                    '''
                    incols = indata.atoms.columns.tolist()
                    outcols = outdata.atoms.columns.tolist()
                    indata.atoms.iloc[i, incols.index("x")] = coords[0]
                    indata.atoms.iloc[i, incols.index("y")] = coords[1]
                    indata.atoms.iloc[i, incols.index("z")] = coords[2]
                    outdata.atoms.iloc[ind, outcols.index("x")] = xyzs[0][0]
                    outdata.atoms.iloc[ind, outcols.index("x")] = xyzs[0][1]
                    outdata.atoms.iloc[ind, outcols.index("x")] = xyzs[0][2]
                    selected_inds.append(ind)

        if mergy_style == 0:
            outdata.atoms = pd.concat([outdata.atoms, indata.atoms])
            outdata.initialization(normalization=True, style=1)
        else:
            if len(selected_inds) > 0:
                outdata.remove_by_inds(selected_inds, style="dyn")
                outdata.initialization(normalization=True, style=1)

        if modify_box:
            outdata.modify_lmpbox(newmatrix, style=style)

        if add_vacuum:
            outdata.add_vacuum(lvac=lvac, direction=direction, zero_coords=True, thres=thres)

        if reset_ids:
            outdata.reset_atom_ids()
        return outdata

    def create_C15_BCC(self, lattpara, nsias, to_center=[0.5, 0.5, 0.5], is_cartesian=False,
                       fC15_0="Z12_0.POSCAR", fC15_1="Z12_1.POSCAR", atom_style="atomic",
                       typesoffset=False, ff_elements=["Fe", "Fe"], atomic_masses=[55.845, 55.845],
                       reset_ids=True):
        """
        :param lattpara: lattice parameter of bcc
        :param nsias: number of SIAs. 2 Self intertitials per C15 Laves phase (2x2x2 of bcc)
        :param to_center: substrate center
        :param is_cartesian: is to_center cartesian
        :param fC15_0: fname for C15 POSCAR (Laves has 8 A sites (shared atoms with BCC) and 16 B sites)
                      4 diamond A sites + 12 (need to delete 4 atom) Bsites = Z-16 polyhedron.
                      Since 4 A sites are shared atoms. The minimum is 12 atoms.
        :param fC15_1:  Due to d3-m symmetry. there is an equivalent C15 which is used to expand C15 clusters
        :param atom_style:
        :param types:
        :param ff_elements:
        :param atomic_masses:
        :param reset_ids:
        :return:
        """

        def get_next_subcen(df_cens, subcen, lattpara):
            if len(df_cens) <= 0:
                print("Increase the cutoff radius for inserting C15 interstitials.")
                raise ValueError("Number of substrate center is zero!")
            n = len(df_cens)
            thissubcen = np.array([df_cens.iloc[n - 1]['x'], df_cens.iloc[n - 1]['y'], df_cens.iloc[n - 1]['z']])
            nor_cens = 2 * (thissubcen - subcen) / lattpara
            nor_cens = np.rint(nor_cens)
            thissum = int(np.sum(nor_cens))
            df_cens.drop(df_cens.tail(1).index, inplace=True)
            #print(f"final nor_cens:{nor_cens} thissubcen:{thissubcen}")
            #print("=========")
            return df_cens, thissubcen, nor_cens

        def transform_tmpdata_coordinates(tmpdata, nor_cens, lattpara):
            rot0 = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]

            rot1 = [[0.0, 0.0, -1.0],
                    [0.0, -1.0, 0.0],
                    [-1.0, 0.0, 0.0]]

            rot2 = [[0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0]]

            rot3 = [[0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0]]

            rots = np.array([rot0, rot1, rot2, rot3])
            nor_cens = np.array(nor_cens).astype(int)
            origin = (0, 0, 0)
            imax = 0
            for idim in range(3):
                m = nor_cens[idim]
                if m >= 0:
                    m = m % 4
                elif m < 0:
                    m = m % -4
                if abs(m) > imax:
                    imax = abs(m)
                nor_cens[idim] = m
            isum = np.sum(nor_cens)
            if isum % 2 == 0:
                if imax > 2:
                    irot = 2
                else:
                    irot = 0
            else:
                if imax > 2:
                    irot = 3
                else:
                    irot = 1

            symmop = SymmOp.from_rotation_and_translation(rots[int(irot)], origin)
            tmpdata.atoms = lmpData.modify_by_symmetry(tmpdata.atoms, symmop, normalization=True)
            return tmpdata

        def remove_atoms(outdata, tmpdata, df_cens, rcut=1.4, remove_C15=False, isubtag="isub"):
            todel = np.array([], dtype=int)
            tmpdel = np.array([], dtype=int)
            s1 = np.array([], dtype=int)
            s2 = np.array([], dtype=int)
            for itmp in range(tmpdata.natoms):
                d = tmpdata.atoms.iloc[itmp].to_dict()
                coords = np.array([d["x"], d["y"], d["z"]])
                inds, xyzs, ds, types = lmpData.compute_site_distance(coords, outdata, rcut=rcut, style=0, sort=True)
                if len(ds) > 0:
                    if remove_C15:
                        if ds[0] < 0.1:
                            tmpdel = np.append(tmpdel, [itmp])
                        else:
                            if len(ds) == 1:
                                if len(s1) > 0:
                                    tmpdel = np.append(tmpdel, [itmp])
                                else:
                                    if len(s2) > 0:
                                        if inds[0] in s2:
                                            s1 = copy.deepcopy(inds)
                                        else:
                                            tmpdel = np.append(tmpdel, [itmp])
                                    else:
                                        s1 = copy.deepcopy(inds)
                            else:
                                if len(s2) > 0:
                                    tmpdel = np.append(tmpdel, [itmp])
                                else:
                                    if len(s1) > 0:
                                        if s1[0] in inds:
                                            s2 = copy.deepcopy(inds)
                                        else:
                                            tmpdel = np.append(tmpdel, [itmp])
                                    else:
                                        s2 = copy.deepcopy(inds)
                    else:
                        todel = np.append(todel, inds)
            if remove_C15:
                todel = copy.deepcopy(s1)
                tmpdel = np.unique(tmpdel)
                print(tmpdel)
                tmpdata.remove_by_inds(tmpdel, style="dyn")
            todel = np.unique(todel)
            isubs = outdata.atoms[isubtag].to_numpy().astype(int)
            isubs = isubs[todel]
            isubcens = df_cens[isubtag].to_numpy().astype(int)
            isubcens = list(isubcens)
            todel_subcens = []
            for i in isubs:
                if i in isubcens:
                    ind = isubcens.index(i)
                    todel_subcens.append(ind)
            todel = np.array(todel).astype(int)
            inds_org = np.arange(len(df_cens), dtype=int)
            goods = np.delete(inds_org, todel_subcens)
            df_cens = df_cens.iloc[goods]
            outdata.remove_by_inds(todel, style="dyn")
            outdata.atoms = pd.concat([outdata.atoms, tmpdata.atoms])
            outdata.initialization(normalization=False, style=1)
            ndelete_sub_this = len(todel)
            ndelete_C15_this = len(tmpdel)
            #print(f"final todel:{todel} tmpdel:{tmpdel}")
            #print("===")
            return outdata, df_cens, ndelete_sub_this, ndelete_C15_this

        outdata = self.deepcopy()
        bondlength = lattpara * np.sqrt(3) / 2.0
        buffer = bondlength * 0.03
        volume = np.power(lattpara, 3)
        isubtag = "isub"
        outdata.insert_atoms_tag(isubtag, np.arange(outdata.natoms, dtype=int))
        if typesoffset:
            print("The C15 have offset type by 1.")
            print("you must input ff_elements and atomic_masses accordingly.")
            outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)
    
        radius = np.power((nsias / 18 + 1) * 8 * volume, 1 / 3)
        inds, xyzs, ds, types = outdata.select_by_radius(radius + bondlength + buffer, center=to_center,
                                                         is_cartesian=is_cartesian, depress=None, style=1, delete=False,
                                                         sort=False)
        isorts = np.argsort(ds)
        xyzs = xyzs[isorts]
        ds = ds[isorts]
        inds = inds[isorts]
        subcen = copy.deepcopy(xyzs[0])
        df_subcens = outdata.atoms.copy(deep=True)
        reversed_inds = inds[::-1]
        df_subcens = df_subcens.iloc[reversed_inds]

        app_scale = lattpara / 3.0
        c15_0 = lmpData.from_POSCAR(fC15_0, atom_style)
        c15_0.insert_atoms_tag(isubtag, np.arange(c15_0.natoms, dtype=int))
        c15_0.scale_data(app_scale)
        c15_0.assert_force_field(ff_elements, atomic_masses=atomic_masses)
        c15_1 = lmpData.from_POSCAR(fC15_1, atom_style)
        c15_1.insert_atoms_tag(isubtag, np.arange(c15_1.natoms, dtype=int))
        c15_1.scale_data(app_scale)
        c15_1.assert_force_field(ff_elements, atomic_masses=atomic_masses)
        laycen = np.array([lattpara, lattpara, lattpara])

        ndelete_sub = 0
        ndelete_C15 = 0
        nadd = 0
        nremains = nsias - nadd
        iloop = 0
        print(f"start nremains:{nremains}")
        while nremains > 0:
            if nremains == 1:
                remove_C15 = True
            else:
                remove_C15 = False
            df_subcens, thissubcen, nor_cens = get_next_subcen(df_subcens, subcen, lattpara)
            thisshift = thissubcen - laycen

            tmpdata = c15_0.deepcopy()
            tmpdata = transform_tmpdata_coordinates(tmpdata, nor_cens, lattpara)

            tmpdata.modify_atoms(translation=thisshift, is_cartesian=True, normalization=False)
            ntmpdata = tmpdata.natoms
            if typesoffset:
                tmpdata.atoms['type'] += 1

            outdata, df_subcens, ndsub_this, ndC15_this = remove_atoms(outdata, tmpdata, df_subcens,
                                                                       rcut=bondlength / 2 + buffer,
                                                                       remove_C15=remove_C15,
                                                                       isubtag=isubtag)
            ndelete_sub += ndsub_this
            ndelete_C15 += ndC15_this
            nadd_this = ntmpdata - ndsub_this - ndC15_this
            nadd += nadd_this
            nremains -= nadd_this
            iloop += 1

            print(f"finished iloop: {iloop}  nremains:{nremains} subcen:{subcen}")
            print(f"thissubcen:{thissubcen} thisshift:{thisshift} norcen:{nor_cens}")
            print(f"nadd_this:{nadd_this} ndsub_this:{ndsub_this} ndC15_this:{ndC15_this}")
            print(f"nadd:{nadd} ndelete_sub:{ndelete_sub} ndelete_C15:{ndelete_C15}")
            print(f"natoms:{outdata.natoms} ncens: {len(df_subcens)}")

            '''
            thisout = outdata.deepcopy()
            thisout.select_by(by="type", vlim=[2, 5], style="INCLUDE", delete=True)
            xs = thisout.atoms["x"].to_numpy()
            ys = thisout.atoms["y"].to_numpy()
            zs = thisout.atoms["z"].to_numpy()
            xyzs = np.vstack((xs, ys, zs))
            xyzmins = np.min(xyzs, axis=1)
            xyzmaxs = np.max(xyzs, axis=1)
            print(f"xyzmins:{xyzmins} xyzmaxs:{xyzmaxs}")
            print(df_subcens.tail())
            '''
            print(f"==========================================")

        if reset_ids:
            outdata.reset_atom_ids()
        return outdata

    def rigid_move(self, nstart, thisshift):
        intz = self.atoms["intz"].to_numpy()
        y = self.atoms["y"].to_numpy()
        y = np.select([intz > nstart, intz <= nstart], [y - thisshift, y])
        self.atoms["y"] = y

    def GPSF_move(self, rshift, dshift, dintlayer, zstart=0, zstart4shift="auto", thres=[0.2, 0.2, 0.2], style=1):
        self.zero_coords(thres=thres)
        intz = np.around(self.atoms["z"]/dintlayer, decimals=0) - zstart
        self.atoms["intz"] = intz.astype(int)
        dhalf = int(max(intz)/2)
        if isinstance(zstart4shift, int):
            zstart4shift = zstart4shift
        else:
            zstart4shift = dhalf - int(rshift/2)

        if style == 1:
            if isinstance(rshift, float) and abs(rshift - int(rshift)) > 0.0001:
                nshift = int(rshift) + 1
                rshifts = []
                for i in range(nshift):
                    if i == 0:
                        rshifts.append(rshift - nshift + 1)
                    else:
                        rshifts.append(1.0)
            else:
                nshift = int(rshift)
                rshifts = []
                for i in range(nshift):
                    if i == 0:
                        rshifts.append(0.5)
                    else:
                        rshifts.append(1.0)
                rshifts.append(0.5)
        else:
            if isinstance(rshift, float) and abs(rshift - int(rshift)) > 0.0001:
                nshift = int(rshift) + 1
                rshifts = []
                for i in range(nshift):
                    if i == nshift - 1:
                        rshifts.append(rshift - nshift + 1)
                    else:
                        rshifts.append(1.0)
            else:
                nshift = int(rshift)
                rshifts = []
                for i in range(nshift):
                    if i == 0:
                        rshifts.append(1.0)
                    else:
                        rshifts.append(1.0)

        for i in range(len(rshifts)):
            nstart = zstart4shift + i
            thisshift = rshifts[i] * dshift
            self.rigid_move(nstart, thisshift)

        newmatrix = copy.deepcopy(self.box.matrix)
        newmatrix[2, 1] =  -rshift * dshift
        self.modify_lmpbox(newmatrix, style=1, reset_ids=True)

    def GSFE_move(self, rshift, dshift, dintlayer, zstart=0, zstart4shift="auto", thres=[0.2, 0.2, 0.2]):
        self.zero_coords(thres=thres)
        intz = np.around(self.atoms["z"] / dintlayer, decimals=0) - zstart
        self.atoms["intz"] = intz.astype(int)
        dhalf = int(max(intz) / 2.0)
        if isinstance(zstart4shift, int):
            zstart4shift = zstart4shift
        else:
            zstart4shift = dhalf
        self.rigid_move(zstart4shift, rshift * dshift)
        newmatrix = copy.deepcopy(self.box.matrix)
        newmatrix[2, 1] =  -rshift * dshift
        self.modify_lmpbox(newmatrix, style=1, reset_ids=True)

    def bcc112_to_omega(self,  a, nz=1, zstart=0, zstart4shift="auto", thres=[0.2, 0.2, 0.2]):
        '''
        self has to be 110, 111, 112 lattice.
        :param nz: # of z layers, one 112 can only hold 2
        :param a: lattice parameter
        :param zstart: starting z layer
        :param zstart4shift: starting y layer for shift
        :return:
        '''
        self.zero_coords(thres=thres)
        dintz = a * np.sqrt(6) / 6.0
        intz = np.around(self.atoms["z"] / dintz, decimals=0) - zstart
        self.atoms["intz"] = intz.astype(int)
        dinty = a * np.sqrt(3) / 6.0
        inty = np.around(self.atoms["y"] / dinty, decimals=0)
        self.atoms["inty"] = inty.astype(int)

        dhalf = int(max(intz) / 2.0)
        if isinstance(zstart4shift, int):
            zstart4shift = zstart4shift
        else:
            zstart4shift = dhalf - 2

        rshift = a * np.sqrt(3) / 12.0
        for i in range(nz):
            thiszstart = zstart4shift + i * 3
            intz = self.atoms["intz"].to_numpy()
            y = self.atoms["y"].to_numpy()
            y = np.select([intz == thiszstart + 2, intz != thiszstart + 2], [y - rshift, y])
            y = np.select([intz == thiszstart + 3, intz != thiszstart + 3], [y + rshift, y])
            self.atoms["y"] = y












