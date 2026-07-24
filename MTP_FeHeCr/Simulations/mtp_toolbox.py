import os,sys
import shutil
import copy
import numpy as np

from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices
from mylammps.inputs.util import generate_rotation_matrix

ff_elements = ["Fe", "He", "Cr"]
atomic_masses = [55.845, 4.0026, 51.9961]
lattice_parameters = [2.83048847, 2.67054960, 2.82958219]
ground_energies = [-8.24159650, 0.01474557, -9.45576250]
Cr_Concs = [0.03, 0.09]

def make_supercell_from_basis(fbasis, supercell, a0=2.83048847, out_atom_style="atomic"):
    outdata = lmpData.from_file(fbasis, "atomic", sort_id=False, parse_velocity=False)
    outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)
    outdata.scale_data(a0, style=0)
    outdata = outdata.make_supercell(supercell)
    if out_atom_style == "molecular":
        outdata.insert_molecular_id()
    return outdata

def Fe2Cr(thisdata, conc, to_typeid=3):
    inds = np.arange(thisdata.natoms, dtype=int)
    thistypes = thisdata.atoms['type'].to_numpy()
    inds_Fe = np.compress(thistypes == 1, inds)
    nFe = len(inds_Fe)
    nCr = int(nFe * conc)
    if nCr > 0:
        np.random.shuffle(inds_Fe)
        inds_Cr = inds_Fe[nFe - nCr: nFe]
        thistypes[inds_Cr] = to_typeid
        thisdata.atoms['type'] = thistypes
    thisdata.get_data_info()
    return thisdata

def Fe2Crs_from_a_file(fname, Cr_concs=Cr_Concs,  foutheader=None, atom_style="atomic"):
    if foutheader is None:
        foutheader = fname.replace(".dat", "")
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)
    outfnames = []
    for i in range(len(Cr_concs)):
        outdata = orgdata.deepcopy()
        c = Cr_concs[i]
        outdata = Fe2Cr(outdata, c)
        outfname = foutheader + "_" + str(int(c*100)) + "Cr.dat"
        outdata.to_file(outfname + ".dat")
        outfnames.append(outfname)
    print(f"-- finished fname:{fname}!")
    return outfnames

def insert_interstitial(fname,  fint, nint, to_typeid=2, atom_style="atomic", atom_style4int="atomic",
                        check_distance=False, rcut=1.5):
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)
    outdata = orgdata.deepcopy()
    data_int = lmpData.from_file(fint, atom_style4int, sort_id=False)
    data_int.assert_force_field(ff_elements, atomic_masses)
    inds_int = np.arange(data_int.natoms, dtype=int)
    np.random.shuffle(inds_int)
    i = 0
    iaccept = 0
    while iaccept < nint:
        nr = inds_int[i]
        indict = data_int.atoms.iloc[nr].to_dict()
        default_dict = outdata.generate_default_dict()
        for key in default_dict.keys():
            if key in indict.keys():
                default_dict[key] = indict[key]

        if check_distance:
            coords = np.array([default_dict['x'], default_dict['y'], default_dict['z']])
            inds, xyzs, distances, types = lmpData.compute_site_distance(coords, outdata,
                                                                         style=0, rcut=rcut, sort=False)
            if len(inds) > 0:
                pass
            else:
                outdata.atoms.loc[outdata.idmax + 1] = default_dict
                outdata.idmax += 1
                outdata.initialization()
                iaccept += 1
        else:
            iaccept += 1
        i += 1
    return outdata

def create_frenkel_pairs(fname, fint, npair, atom_style="atomic", atom_style4int="atomic"):
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)
    outdata = orgdata.deepcopy()
    inds = np.arange(outdata.natoms, dtype=int)
    np.random.shuffle(inds)
    del_inds = inds[0:npair]
    outdata.remove_by_inds(del_inds, style="dyn")
    outdata.to_file("tmp.dat")
    outdata = insert_interstitial("tmp.dat", fint, npair, to_typeid=1,
                                  atom_style=atom_style, atom_style4int=atom_style4int, rcut=1.5)
    return outdata


def make_screw_dislocation4relax(supercell, a0=2.83048847, fbasis="bcc110_112_111_orth4screw.data", atom_style="atomic"):
    outdata = make_supercell_from_basis(fbasis, supercell, a0=a0, out_atom_style="atomic")
    burgerm = np.sqrt(3) * a0 / 2.0
    orientation = False
    add_vacuum = False
    direction = 0
    outdata.create_screw_dislocation(burgerm, 1, style="bcc", handle_pbc="tilt",
                                     orientation=orientation,
                                     add_vacuum=add_vacuum, direction=direction)
    return outdata

def add_vacuum_relaxed_screw(fname, lvac, atom_style="atomic", orientation=True):
    outdata = lmpData.from_file(fname, atom_style)
    direction = 0
    outdata.add_vacuum(direction=direction, lvac=lvac, thres=[0.3, 0.3, 0.3])
    if orientation:
        newaxis = [2, 1, 0]
        outdata.swap_axes(newaxis)
    return outdata

def create_HenVm(outdata, data_int, nHe, mV, inds_vac, inds_int):
    indicts_vac = []
    for i in range(len(inds_vac)):
        indicts_vac.append(outdata.atoms.iloc[inds_vac[i]].to_dict())
    indicts_int = []
    for i in range(len(inds_int)):
        indicts_int.append(data_int.atoms.iloc[inds_int[i]].to_dict())

    if mV > 0:
        outdata.remove_by_inds(inds_vac[0:mV], style="dyn")
    if nHe > 0:
        if nHe <= mV:
            for i in range(nHe):
                indict = indicts_vac[i]
                outdata.add_an_entry(indict, loc=None, check_distance=False)
        else:
            for i in range(mV):
                indict = indicts_vac[i]
                outdata.add_an_entry(indict, loc=None, check_distance=False)
            nadd = nHe - mV
            for i in range(nadd):
                indict = indicts_int[i]
                outdata.add_an_entry(indict, loc=None, check_distance=False, rcut=1.0)
    return outdata

def create_HenVms_from_basis(supercell, fbasis, nHes, mVs, fbasis_int="tetra_bcc.data", a0=2.83048847):
    refdata = make_supercell_from_basis(fbasis, supercell, a0=a0, out_atom_style="atomic")
    data_int = make_supercell_from_basis(fbasis_int, supercell, a0=a0, out_atom_style="atomic")
    radius = a0 + 0.1
    inds, xyzs, ds, types = refdata.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)

    inds_vac = [0, 1]
    xyz0 = xyzs[0]
    xyz1 = xyzs[1]
    diffs = xyz1 - xyz0
    for i in range(2, len(inds)):
        xyz = xyzs[i]
        if xyz[1] == xyz1[1] and xyz[2] == xyz1[2]:
            inds_vac.append(i)
        elif xyz[0] == xyz0[0] and xyz[1] == xyz0[1] and (xyz[2] - xyz0[2]) * diffs[2] > 0:
            inds_vac.append(i)
        else:
            continue
    inds_vac = np.array(inds_vac)
    xyzs_vac = xyzs[inds_vac]
    inds_vac = inds[inds_vac]
    for i in range(len(xyzs_vac)):
        print(f"-- vacancy site {i} is at {xyzs_vac[i]}!")

    inds, xyzs, ds, types = data_int.select_by_radius(radius, depress=None, center=[0.5, 0.5, 0.5],
                                                  is_cartesian=False, delete=False, sort=True)
    inds_int = []
    for i in range(0, len(inds)):
        xyz = xyzs[i]
        diff = xyz - xyz1 #xyz1 is reference coordinates in vacancy site (2nd)
        d = np.linalg.norm(diff)
        if d < a0 * np.sqrt(5) / 4.0 + 0.1:
            inds_int.append(i)
        else:
            continue
    inds_int = np.array(inds_int)
    xyzs_int = xyzs[inds_int]
    inds_int = inds[inds_int]
    for i in range(len(xyzs_int)):
        print(f"-- interstitial site {i} is at {xyzs_int[i]}!")

    for n in range(len(nHes)):
        nHe = nHes[n]
        for m in range(len(mVs)):
            mV = mVs[m]
            outdata = refdata.deepcopy()
            outdata = create_HenVm(outdata, data_int, nHe, mV, inds_vac, inds_int)
            fname = "Fe_He" + str(nHe) + "V" + str(mV) + ".dat"
            outdata.to_file(fname)
            print(f"-- finished fname:{fname}!")
    return outdata



