import os,sys
import shutil
import copy
import numpy as np
import pandas as pd

from pymatgen.core.lattice import Lattice
from sympy.codegen.ast import continue_

from mylammps.inputs.data import lmpBox, lmpData, find_symmop_lattices


ff_elements = ["Fe", "He", "Cr"]
atomic_masses = [55.845, 4.0026, 51.9961]
lattice_parameters = [2.83048847, 2.67054960, 2.82958219]
ground_energies = [-8.24159650, 0.01474557, -9.45576250]
Cr_Concs = [0.03, 0.10]
a0Crs = [2.83218, 2.83511]
epaCrs =[-8.27616063, -8.35194938]
epaCrs_partition = [[-7.0433014175257735, -48.138606458333335], [-4.280546540747361, -44.994575187500004]]

def sort_by_dynamic_center(xyzs):
    xyzs = np.array(xyzs)
    if xyzs.ndim == 1:
        raise ValueError("xyzs should be a 2D array!")

    shape = xyzs.shape

    org_xyzs = copy.deepcopy(xyzs)
    inds = np.arange(shape[0], dtype=int)
    thisxyzs = np.empty((0,3), dtype=float)
    inds_sorted = []

    # 0 is the first atom
    thisxyzs = np.vstack((thisxyzs, org_xyzs[0]))
    org_xyzs = np.delete(org_xyzs, 0, axis=0)
    inds_sorted.append(0)
    inds = np.delete(inds, 0, axis=0)
    center = np.mean(thisxyzs, axis=0)
    iloop = 0
    while len(inds) > 0:
        ds = []
        for i in range(len(inds)):
            d = np.linalg.norm(org_xyzs[i] - center)
            ds.append(d)
        ds = np.array(ds)
        argmin = np.argmin(ds)
        global_ind = inds[argmin]
        thisxyzs = np.vstack((thisxyzs, org_xyzs[argmin]))
        org_xyzs = np.delete(org_xyzs, argmin, axis=0)
        inds_sorted.append(global_ind)
        inds = np.delete(inds, argmin, axis=0)
        center = np.mean(thisxyzs, axis=0)
        iloop += 1
    return np.array(inds_sorted)


def sort_by_ref_coords(xyzs, ref_xyzs, max_ref=4):
    ref_xyzs = np.array(ref_xyzs)[0:max_ref, :]
    df = pd.DataFrame(xyzs, columns=["x", "y", "z"])
    ds = []
    for i in range(len(xyzs)):
        dsi = []
        for j in range(len(ref_xyzs)):
            d = np.linalg.norm(xyzs[i] - ref_xyzs[j])
            dsi.append(d)
        ds.append(dsi)

    sort_keys=[]
    ds = np.array(ds).T
    for j in range(len(ref_xyzs)):
        key = "d_" + str(j)
        df[key] = ds[j]
        sort_keys.append(key)
    df = df.sort_values(by=sort_keys)
    sorted_inds = df.index.to_numpy()
    return sorted_inds

def make_supercell_from_basis(fbasis, supercell, a0=2.83048847, out_atom_style="atomic"):
    outdata = lmpData.from_file(fbasis, "atomic", sort_id=False, parse_velocity=False)
    outdata = outdata.make_supercell(supercell)
    outdata.scale_data(a0, style=0)
    outdata.assert_force_field(ff_elements, atomic_masses=atomic_masses)
    if out_atom_style == "molecular":
        outdata.insert_molecular_id()
    return outdata

def check_Cr_conc(thisdata, conc):
    inds = np.arange(thisdata.natoms, dtype=int)
    thistypes = thisdata.atoms['type'].to_numpy()
    inds_Fe = np.compress(thistypes == 1, inds)
    inds_Cr = np.compress(thistypes == 3, inds)

    nFe = len(inds_Fe)
    nCr = len(inds_Cr)
    nCr_conc = int((nFe + nCr) * conc)
    nCr_diff = nCr_conc - nCr
    if nCr_diff > 0:
        np.random.shuffle(inds_Fe)
        inds_Cr = np.append(inds_Cr, inds_Fe[nFe - nCr_diff: nFe])
        thistypes[inds_Cr] = 3
        thisdata.atoms['type'] = thistypes
    elif nCr_diff < 0:
        np.random.shuffle(inds_Cr)
        inds_Fe = np.append(inds_Fe, inds_Cr[nCr - nCr_diff: nCr])
        thistypes[inds_Fe] = 1
        thisdata.atoms['type'] = thistypes
    thisdata.get_data_info()
    return thisdata

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
        outdata.to_file(outfname )
        outfnames.append(outfname)
    print(f"-- finished fname:{fname}!")
    return outfnames

def insert_inter_from_fint(fname,  fint, nint, to_typeid=2, atom_style="atomic", atom_style4int="atomic",
                        check_distance=False, rcut=1.5):
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)
    print(f"-- natoms original:{orgdata.natoms}!")
    outdata = orgdata.deepcopy()
    data_int = lmpData.from_file(fint, atom_style4int, sort_id=False)
    data_int.assert_force_field(ff_elements, atomic_masses)
    inds_int = np.arange(data_int.natoms, dtype=int)
    np.random.shuffle(inds_int)
    iloop = 0
    iaccept = 0
    while iaccept < nint:
        nr = inds_int[iloop]
        indict = data_int.atoms.iloc[nr].to_dict()
        default_dict = outdata.generate_default_dict()
        for key in default_dict.keys():
            if key in indict.keys():
                default_dict[key] = indict[key]
        default_dict['type'] = to_typeid

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
        iloop += 1

    print(f"-- natoms after insertion:{outdata.natoms}!")
    print(f"--- finished fname:{fname} ---")
    return outdata

def create_frenkel_pairs(fname, fint, npair, atom_style="atomic", atom_style4int="atomic",
                         check_distance=False, rcut=1.5):
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)

    outdata = orgdata.deepcopy()
    inds = np.arange(outdata.natoms, dtype=int)
    np.random.shuffle(inds)
    del_inds = inds[0:npair]
    outdata.remove_by_inds(del_inds, style="dyn")
    outdata.to_file("tmp.dat")
    outdata = insert_inter_from_fint("tmp.dat", fint, npair, to_typeid=1,
                                  atom_style=atom_style, atom_style4int=atom_style4int,
                                  check_distance=check_distance, rcut=rcut)
    print(f"--- finished fname:{fname} ---")
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

def create_dislocation_loop_basis(supercell, supercell4layer, radius, center=[0.5, 0.5, 0.5], a0=2.83048847,
                                  fbasis=None, fbasisid=0,
                                  flayer=None, flayerid=0,
                                  mergy_style=0, loop_shape="circle",
                                  splits=[0.0, -0.6, -0.6], lengths=None, subdata=None,):
    '''
    layms = [
        np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]),
        np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]]),
        np.array([[2, 1, 1], [-1, 1, 1], [0, 1, -1]])
    ]

    centers = [[0.5, 0.5, 0.25], [0.5, 0.5, 0.75]]
    splitss = [[-0.15, -0.1, -0.3], [0, -0.23, -0.23]] wrt a0

    100 in 100 splits [-0.25, -0.5, 0.0]
    111 in 111 splits [0.0, -0.25, -0.25]

    note that the second SIL has the identical orientation with substrate.
    this second splits on the second SIL, visually has 111/2 but has higher energy
    '''

    if fbasisid == 0:
        if fbasis is None:
            fbasis = "bcc112_111_110.data"
        subm = np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]])
    else:
        if fbasis is None:
            fbasis = "bcc.data"
        subm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if flayerid == 0:
        if flayer is None:
            flayer = "bcc112_111_110.data"
        burgerm = a0 * np.sqrt(3) / 2
        bondlength = a0 * np.sqrt(3) / 2
        layerm = np.array([[1, 1, -2], [1, 1, 1], [1, -1, 0]])
    else:
        if flayer is None:
            flayer= "bcc.data"
        layerm = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        burgerm = a0
        bondlength = a0 * np.sqrt(3) / 2
    if subdata is None:
         subdata = make_supercell_from_basis(fbasis, supercell, a0=a0, out_atom_style="atomic")
    layerdata = make_supercell_from_basis(flayer, supercell4layer, a0=a0, out_atom_style="atomic")


    rcut = 0.2 * a0
    tolerance = 0.1 * a0
    depress = 1
    check_distance = True

    symmop = find_symmop_lattices(Lattice(subm), Lattice(layerm))
    laycen = np.array([0.5, 0.5, 0.5])
    laycen = np.dot(laycen.T, layerdata.box.matrix)
    laycen = laycen.T
    if flayerid == 0:
        laycen[depress] = bondlength / 3
    else:
        laycen[depress] = bondlength / 2

    ind, laycen, d, itypes = layerdata.find_center_atom_coords(bondlength + 0.1,
                                                             center=laycen, is_cartesian=True, style=1)
    if loop_shape == "circle":
        layerdata.select_by_radius(radius, center=laycen, is_cartesian=True, depress=depress, delete=True, style=1)
    else:
        if lengths is None:
            lengths = [radius * 2.0] * 3
        layerdata.select_by_coords_wrt_center(lengths, center=laycen, is_cartesian=True,
                                              depress=depress, delete=True, style=1)

    layerdata.atoms = lmpData.modify_by_symmetry(layerdata.atoms, symmop, normalization=True)
    layerdata.to_file("layer.dat")
    outdata = subdata.merge_data_with_splits(layerdata, bondlength,
                                             mergy_style=mergy_style,
                                             to_center=center, is_cartesian=False,
                                             splits=splits, rcut=rcut, tolerance=tolerance,
                                             modify_box=False, newmatrix=None, style=0,
                                             add_vacuum=False, lvac=20.0, direction=2,
                                             reset_ids=True)


    print(f"finished radius:{radius} ")
    print("=====================")
    if mergy_style == 0:
        fout = "SIL_"
    else:
        fout = "VL_"
    if flayerid == 0:
        fout += "111_"
    else:
        fout += "100_"
    if fbasisid == 0:
        fout += "111_"
    else:
        fout += "100_"
    fout += str(loop_shape[0:3]) + "_d_" +  str(int(radius * 2.0)) + "A.dat"
    outdata.to_file(fout)
    return outdata

def create_HenVm(outdata, data_int, nHe, mV, inds_vac, inds_int, to_typeid=2, check_distance=False, rcut=1.2):
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
                indict['type'] = to_typeid
                outdata.add_an_entry(indict, loc=None, check_distance=False)
        else:
            for i in range(mV):
                indict = indicts_vac[i]
                indict['type'] = to_typeid
                outdata.add_an_entry(indict, loc=None, check_distance=False)

            nadd = nHe - mV
            if check_distance:
                iloop = 0
                iaccept = 0
                while iaccept < nadd and iloop < len(inds_int):
                    indict = indicts_int[iloop]
                    coords = np.array([indict['x'], indict['y'], indict['z']])
                    inds, xyzs, distances, types = lmpData.compute_site_distance(coords, outdata,
                                                                            style=0, rcut=rcut, sort=False)
                    if len(inds) > 0:
                        pass
                        # print(f"iloop:{iloop} coords:{coords} xyzs:{xyzs[0]} id:{inds[0]} distance:{distances[0]}")
                    else:
                        outdata.atoms.loc[outdata.idmax + 1] = indict
                        outdata.idmax += 1
                        outdata.initialization()
                        iaccept += 1
                    iloop += 1
                # print(f"nHe:{nHe} mV:{mV} nadd:{nadd} length:{len(inds_int)} iaccept:{iaccept} iloop:{iloop}")
            else:
                for i in range(nadd):
                    indict = indicts_int[i]
                    indict['type'] = to_typeid
                    outdata.add_an_entry(indict, loc=None, check_distance=False, rcut=1.0)

    outdata.reset_atom_ids()
    return outdata

def create_dumbbell(fname, ind, dumbbelltype=0, to_typeid=3, atom_style="atomic", to_typeid4org=None,
                        a0=2.83048847):
    '''

    :param fname:
    :param ind:
    :param dumbbelltype: 1:100 2:111 0:110
    :param to_typeid:
    :param atom_style:
    :param to_typeid4org:
    :param a0:
    :return:
    '''
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(ff_elements, atomic_masses)
    outdata = orgdata.deepcopy()

    tol = 0.1 * a0
    radius = a0 + tol
    bondlength = a0 * np.sqrt(3) / 2

    center_dict = outdata.atoms.iloc[ind].to_dict()
    center = np.array([center_dict['x'], center_dict['y'], center_dict['z']])
    inds, xyzs, ds, types = outdata.select_by_radius(radius, depress=None, center=center,
                                                     is_cartesian=True, style=0,
                                                     delete=False, sort=True)

    int_dict = copy.deepcopy(center_dict)
    xyzorg = xyzs[0]
    if dumbbelltype == 1:
        for i in range(len(inds)-1, -1, -1):
            d = ds[i]
            if d > a0 - tol and d < a0 + tol:
                indint = inds[i]
                xyzint = xyzs[i]
                diffs = (xyzint - xyzorg) * 0.33
                for j in range(2, 3):
                    center_dict['j'] = xyzorg[j] + diffs[j]
                    int_dict['j'] = xyzorg[j] - diffs[j]
                break
    elif dumbbelltype == 2:
        for i in range(len(inds)-1, -1, -1):
            d = ds[i]
            if d > a0 - tol and d < a0 + tol:
                indint = inds[i]
                xyzint = xyzs[i]
                diffs = (xyzint - xyzorg) * 0.25
                for j in range(3):
                    center_dict['j'] = xyzorg[j] + diffs[j]
                    int_dict['j'] = xyzorg[j] - diffs[j]
                break
    else:
        for i in range(len(inds)-1, -1, -1):
            d = ds[i]
            if d > a0 - tol and d < a0 + tol:
                indint = inds[i]
                xyzint = xyzs[i]
                diffs = (xyzint - xyzorg) * 0.25
                for j in range(1,3):
                    center_dict['j'] = xyzorg[j] + diffs[j]
                    int_dict['j'] = xyzorg[j] - diffs[j]
                break
    if isinstance(to_typeid4org, int):
        center_dict['type'] = to_typeid4org
    outdata.atoms.iloc[ind] = center_dict
    int_dict['type'] = to_typeid
    outdata.add_an_entry(int_dict, loc=None, check_distance=False)
    return outdata

def create_HenVms_from_basis(supercell, fbasis, nHes, mVs,
                             fbasis_int="tetra_bcc.data", supercell4int=None, a0=2.83048847,
                             center=[0.5, 0.5, 0.5], is_cartesian=False, style=0,
                             refdata=None, outfheader="Fe", check_distance=False, rcut=1.2):
    '''
    :param supercell:
    :param fbasis:
    :param nHes:
    :param mVs:
    :param fbasis_int:
    :param a0:
    :param center:
    :return:
    '''
    if refdata is None:
        refdata = make_supercell_from_basis(fbasis, supercell, a0=a0, out_atom_style="atomic")

    if supercell4int is None:
        supercell4int = copy.deepcopy(supercell)

    tol = 0.1 * a0
    radius = a0 + tol
    inds, xyzs, ds, types = refdata.select_by_radius(radius, depress=None, center=center,
                                                     is_cartesian=is_cartesian, style=style,
                                                     delete=False, sort=True)
    inds_vac = sort_by_dynamic_center(xyzs) #local index
    xyzs_vac = xyzs[inds_vac]
    inds_vac = inds[inds_vac]


    data_int = make_supercell_from_basis(fbasis_int, supercell4int, a0=a0, out_atom_style="atomic")
    if not is_cartesian:
        refcenter = np.dot(center, refdata.box.matrix)
    else:
        refcenter = copy.deepcopy(center)
    intcent = np.dot([0.5, 0.5, 0.5], data_int.box.matrix)
    translations = refcenter - intcent


    data_int.modify_atoms(translation=translations, is_cartesian=True, normalization=False)
    inds, xyzs, ds, types = data_int.select_by_radius(radius, depress=None, center=refcenter,
                                                      is_cartesian=True, style=0,
                                                      delete=False, sort=True)
    inds_int = sort_by_ref_coords(xyzs, xyzs_vac)
    xyzs_int = xyzs[inds_int]
    inds_int = inds[inds_int]

    outfiles = []
    for n in range(len(nHes)):
        nHe = nHes[n]
        for m in range(len(mVs)):
            mV = mVs[m]
            if nHe == 0 and mV == 0:
                pass
            else:
                outdata = refdata.deepcopy()
                outdata = create_HenVm(outdata, data_int, nHe, mV, inds_vac, inds_int,
                                       check_distance=check_distance, rcut=rcut)
                fname = outfheader + "_He" + str(nHe) + "V" + str(mV) + ".dat"
                outdata.to_file(fname)
                outfiles.append(fname)
                print(f"-- finished fname:{fname}!")
    return outfiles





