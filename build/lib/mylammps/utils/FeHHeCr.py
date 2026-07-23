import os,sys
import numpy as np
import copy
import itertools
from myVASP.inputs.inputs import myPOSCAR
from mylammps.inputs.data import lmpBox, lmpData

myElements = ["Fe", "H", "He", "Cr"]
myAtomicMasses = [55.845, 1.008, 4.003, 51.996]

def get_data_files(infold, app=".dat"):
    fnames = []
    n = len(app)
    for fname in os.listdir(infold):
        if len(fname) > n:
            if fname[len(fname)-n:len(fname)] == app:
                fnames.append(fname)
    fnames.sort()
    return fnames

def set_2to3(thisdata):
    inds = np.arange(thisdata.natoms, dtype=int)
    thistypes = thisdata.atoms['type'].to_numpy()
    inds_2 = np.compress(thistypes == 2, inds)
    if len(inds_2) > 0:
        thistypes[inds_2] = 3
        thisdata.atoms['type'] = thistypes
    thisdata.get_data_info()
    return thisdata

def set_types2lmp(thisPOS, thisdata,
                  ff_elements=myElements, atomic_masses=myAtomicMasses):
    types = []
    for i in range(thisPOS.natoms):
        thissym = thisPOS.site_symbols[i]
        if thissym == "Mn":
            thissym = "Cr"
        try:
            thistype = ff_elements.index(thissym) + 1
        except:
            thistype = len(ff_elements) + 1
        types.append(thistype)
    thisdata.atoms["type"] = types
    return thisdata

def Fe2Cr(thisdata, conc, to_typeid=4):
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

def Fe2Crs_from_a_file(fname, Cr_concs = [0.04, 0.08, 0.12], foutheader=None, atom_style="atomic"):
    if foutheader is None:
        foutheader = fname.replace(".dat", "")
    orgdata = lmpData.from_file(fname, atom_style, sort_id=False)
    orgdata.assert_force_field(myElements, myAtomicMasses)
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

def vasp2lmp(fname, fout, supercell=[1,1,1], selective_dynamics=False,
             atom_style="atomic", set_types=True, scale=2.83037145, normalization=False,
             ff_elements=myElements, atomic_masses=myAtomicMasses):
    thisPOS = myPOSCAR.from_file(fname)
    thisPOS.make_supercell(supercell)
    thisPOS.to_lmpdata(fout)
    thisdata = lmpData.from_file(fout, atom_style, sort_id=False, parse_velocity=False)
    thisdata.assert_force_field(ff_elements, atomic_masses)
    nfixed = 0
    if selective_dynamics:
        sds = [site.selective_dynamics[2] for site in thisPOS.struct]
        sds = np.array(sds)
        inds_fixed = np.compress(sds == 0, sds)
        inds = np.argsort(sds)
        thisdata.atoms = thisdata.atoms.iloc[inds]
        thisdata.reset_atom_ids()
        nfixed = len(inds_fixed)
        print(f"number of fixed atoms in {fname} is {nfixed}")
    if set_types:
        thisdata = set_types2lmp(thisPOS, thisdata, ff_elements=ff_elements, atomic_masses=atomic_masses)

    if normalization:
        thisdata.scale_data(1 / scale, style=0)
    thisdata.to_file(fout)
    return thisdata, nfixed


