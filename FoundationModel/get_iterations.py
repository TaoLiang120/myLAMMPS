import os
import numpy as np
import copy
import pandas as pd
import itertools
from numpy import pi

from mylammps.inputs.data import lmpData, lmpBox

Pots = ["FeHe_Gao"]
Labels = ["Gao"]
PDs = ["1V1He"]
'''
Pots = ["M07_eam.fs"]
Labels = ["M07"]
PDs = ["dumbbell011"]
'''
nsupers = [20.0]
a1_ref = 2.75590864
a2_ref = 2.83037145
app_scale = 0.9736915063591047
a1_ref = a2_ref * app_scale
a_refs = [a1_ref, a2_ref]
#a_refs = [a1_ref]
idsps = [11]  ## for Gao's 1V1He
foldhead = "11"

'''
idsps = [6] ## for M07's dumbbell011
idsps = [9]
foldhead = "2"
'''
#idsps = None

def get_lattice(label, pot):
    fname = "Grounds_summary_Group_" + label + ".csv"
    print(fname)
    df = pd.read_csv(fname)
    latt = df.at[0, "Lattice constant"]
    return latt

def spdata2data(grounddata, spdata):
    thisdata = grounddata.deepcopy()
    inds = spdata.atoms.index.to_numpy()
    for ind in inds:
        thisdata.atoms.loc[ind] = spdata.atoms.loc[ind]
    thisdata.coords2fracts()
    return thisdata

def disps2data(avdata, df, df_local):
    thisdata = avdata.deepcopy()
    ndf = len(df)
    ndf0 = len(df_local)
    nthis = min(ndf, ndf0)
    dispx = df['x'].to_numpy()[0:nthis] - df_local['x'].to_numpy()[0:nthis]
    dispy = df['y'].to_numpy()[0:nthis] - df_local['y'].to_numpy()[0:nthis]
    dispz = df['z'].to_numpy()[0:nthis] - df_local['z'].to_numpy()[0:nthis]
    if thisdata.natoms < nthis:
        thisdata.atoms['x'] = thisdata.atoms['x'].to_numpy() + dispx[0:thisdata.natoms]
        thisdata.atoms['y'] = thisdata.atoms['y'].to_numpy() + dispy[0:thisdata.natoms]
        thisdata.atoms['z'] = thisdata.atoms['z'].to_numpy() + dispz[0:thisdata.natoms]
    elif thisdata.natoms > nthis:
        thisdata.atoms['x'] = thisdata.atoms['x'].to_numpy() + np.append(dispx, np.zeros(thisdata.natoms-nthis))
        thisdata.atoms['y'] = thisdata.atoms['y'].to_numpy() + np.append(dispy, np.zeros(thisdata.natoms-nthis))
        thisdata.atoms['z'] = thisdata.atoms['z'].to_numpy() + np.append(dispz, np.zeros(thisdata.natoms-nthis))
    else:
        thisdata.atoms['x'] = thisdata.atoms['x'].to_numpy() + dispx
        thisdata.atoms['y'] = thisdata.atoms['y'].to_numpy() + dispy
        thisdata.atoms['z'] = thisdata.atoms['z'].to_numpy() + dispz

    maxx  = np.max(np.absolute(dispx))
    maxy  = np.max(np.absolute(dispy))
    maxz  = np.max(np.absolute(dispz))
    displmax = [maxx, maxy, maxz]
    print(f"displmax:{displmax}")
    thisdata.coords2fracts()
    return thisdata

def get_data_DataOut(pot, pdt):
    fpath = os.path.join(pot, pdt, 'DataOut')
    fdata = os.path.join(fpath, "KMC_0.dat")
    gdata = lmpData.from_file(fdata, "molecular")
    for f in os.listdir(fpath):
        if "KMC_0_Data_SP" in f:
            fsp = f
            break
    fname = os.path.join(fpath, fsp)
    spdata = lmpData.from_file(fname, "molecular")
    idsp = fsp.replace("KMC_0_Data_SP_", "")
    idsp = idsp.replace(".dat", "")
    idsp = int(idsp)
    try:
        fdata = os.path.join(pot, pdt, "AVOut", "KMC_0_AV_0.dat")
        avdata = lmpData.from_file(fdata, "molecular")
    except:
        errorlog = "Unable to find avdata!"
        raise ValueError(errorlog)
    return gdata, avdata, spdata, idsp

def get_df_coords(pot, pdt, idsp):
    fpath = os.path.join(pot, pdt, 'ITERATION_RESULTS', str(0)+"_"+str(idsp))
    fnames = []
    itrans = []
    iteras = []
    for f in os.listdir(fpath):
        if "Coords_" in f:
            fhead = f.replace(".csv", "")
            fhead = fhead.replace("Coords_","")
            fhead = fhead.split("_")
            itran = int(fhead[0])
            itera = int(fhead[1])
            if itran in itrans:
                ind = itrans.index(itran)
                iteraold = iteras[ind]
                if itera < iteraold:
                    iteras[ind]=itera
                    fnames[ind]=f
            else:
                fnames.append(f)
                itrans.append(itran)
                iteras.append(itera)
    itrans = np.array(itrans)
    iteras = np.array(iteras)
    fnames = np.array(fnames)
    inds = np.argsort(itrans)
    fnames = fnames[inds]
    iteras = iteras[inds]
    itrans = itrans[inds]

    df_coords = []
    for idf in range(len(fnames)):
        fname = fnames[idf]
        df = pd.read_csv(os.path.join(fpath, fname))
        df_coords.append(df)
        if idf == 0:
            print(fname)
            print("000")
    return df_coords

def get_displmax(thisdata, data0):
    x = thisdata.atoms['x'].to_numpy() - data0.atoms['x'].to_numpy()
    y = thisdata.atoms['y'].to_numpy() - data0.atoms['y'].to_numpy()
    z = thisdata.atoms['z'].to_numpy() - data0.atoms['z'].to_numpy()
    maxx = np.max(np.absolute(x))
    maxy = np.max(np.absolute(y))
    maxz = np.max(np.absolute(z))
    return [maxx, maxy, maxz]

def distance_check(thisdata):
    rmin = 10
    for i in range(thisdata.natoms):
        d = thisdata.atoms.iloc[i].to_dict()
        coords = np.array([d["x"], d["y"], d["z"]])
        inds, xyzs, ds, types = lmpData.compute_site_distance(coords, thisdata,
                                                                  rcut=3, style=0, sort=True)
        if ds[1] < rmin:
            rmin = ds[1]
    return rmin


def Get_INDs(indata, ref_data, nsuper, thislatt, a0=a1_ref, nsuper_vasp=4):
    thisdata = indata.deepcopy()
    thisdata.scale_data(a0/thislatt, style=0)
    s = -(nsuper - nsuper_vasp) * 0.5 * a0
    shift = [s, s, s]
    thisdata.modify_atoms(translation=shift, inds=None, is_cartesian=True,
                 normalization=False)
    ids = thisdata.atoms.index.tolist()
    goods = []
    for i in range(ref_data.natoms):
        d = ref_data.atoms.iloc[i].to_dict()
        coords = np.array([d["x"], d["y"], d["z"]])
        inds, xyzs, ds, types = lmpData.compute_site_distance(coords, thisdata,
                                                 rcut=a0/1.5, style=1, sort=True)

        if len(ds) > 0:
            ind = inds[0]
            iind = ids[ind]
            if iind in goods:
                print(f"WARING i:{i} iind:{iind} inds:{inds}")
            else:
                goods.append(iind)
    print(len(goods))
    goods = np.array(goods).astype(int)
    return goods


def chop2vasp(indata, centers, nsuper, thislatt,
        a0=a1_ref, INDs=None, nsuper_vasp=None):
    thisdata = indata.deepcopy()
    thisdata.scale_data(a0/thislatt, style=0)
    cens = centers  * a0 / thislatt
    nbuff = a0 / 4.0
    x0 = cens[0] - 0.5 * nsuper_vasp * a0 - nbuff
    x1 = x0 + nsuper_vasp * a0
    y0 = cens[1] - 0.5 * nsuper_vasp * a0 - nbuff
    y1 = y0 + nsuper_vasp * a0
    z0 = cens[2] - 0.5 * nsuper_vasp * a0 - nbuff
    z1 = z0 + nsuper_vasp * a0

    s = -(nsuper - nsuper_vasp) * 0.5 * a0
    shift = [s, s, s]
    if INDs is None:
        thisdata.select_by_coords(xlim=[x0, x1], ylim=[y0, y1], zlim=[z0, z1],
                         Fractional=False, style="INCLUDE", delete=True)
    else:
        thisdata.atoms = thisdata.atoms.loc[INDs]
        thisdata.initialization()
    thisdata.modify_atoms(translation=shift, inds=None, is_cartesian=True,
                 normalization=False)
    #thisdata.set_coord_mins_to_zeros()
    newmatrix = np.eye(3) * nsuper_vasp * a0
    thisdata.modify_lmpbox(newmatrix, style=1, reset_ids=False)
    
    #thisdata.normalize_coords()
    thisdata.initialization()
    INDs = thisdata.atoms.index.to_numpy()
    thisdata.remove_molecular_id()
    #thisdata.reset_atom_ids()
    return thisdata, INDs

def generate_empty_avdata(indata):
    thisdata = indata.deepcopy()
    thisdata.reset_atom_ids()
    zeros = np.zeros(thisdata.natoms)
    thisdata.atoms['x'] = zeros
    thisdata.atoms['y'] = zeros
    thisdata.atoms['z'] = zeros
    thisdata.coords2fracts()
    return thisdata

def df2data(zerodata, df):
    thisdata = zerodata.deepcopy()
    ndf = len(df)
    if thisdata.natoms < ndf:
        thisdata.atoms['x'] = df['x'].to_numpy()[0:ndf]
        thisdata.atoms['y'] = df['y'].to_numpy()[0:ndf]
        thisdata.atoms['z'] = df['z'].to_numpy()[0:ndf]
    elif thisdata.natoms > ndf:
        thisdata.atoms = thisdata.atoms.loc[np.arange(ndf, dtype=int)+1]
        thisdata.natoms = len(thisdata.atoms)
        thisdata.atoms['x'] = df['x'].to_numpy()
        thisdata.atoms['y'] = df['y'].to_numpy()
        thisdata.atoms['z'] = df['z'].to_numpy()
    else:
        thisdata.atoms['x'] = df['x'].to_numpy()
        thisdata.atoms['y'] = df['y'].to_numpy()
        thisdata.atoms['z'] = df['z'].to_numpy()
    thisdata.coords2fracts()
    return thisdata

def data2vasp(indata, centers, nsuper, thislatt, a0=2.83, INDs=None, nsuper_vasp=4.0):
    thisdata = indata.deepcopy()
    thisdata.scale_data(a0 / thislatt, style=0)
    cens = centers * a0 / thislatt
    nbuff = a0 / 4.0
    x0 = cens[0] - 0.5 * nsuper_vasp * a0 - nbuff
    x1 = x0 + nsuper_vasp * a0
    y0 = cens[1] - 0.5 * nsuper_vasp * a0 - nbuff
    y1 = y0 + nsuper_vasp * a0
    z0 = cens[2] - 0.5 * nsuper_vasp * a0 - nbuff
    z1 = z0 + nsuper_vasp * a0
    s = -(nsuper - nsuper_vasp) * 0.5 * a0
    shift = [s, s, s]
    print(thisdata.atoms.head())
    print("111")
    if INDs is None:
        thisdata.select_by_coords(xlim=[x0, x1], ylim=[y0, y1], zlim=[z0, z1],
                                  Fractional=False, style="INCLUDE", delete=True)
    else:
        thisdata.atoms = thisdata.atoms.loc[INDs]
        thisdata.initialization()
    #thisdata.modify_atoms(translation=shift, inds=None, is_cartesian=True,
    #                      normalization=False)

    newmatrix = np.eye(3) * nsuper_vasp * a0
    thisdata.modify_lmpbox(newmatrix, style=1, reset_ids=False)
    print(thisdata.atoms.head())
    print("222")
    thisdata.initialization()
    print(thisdata.atoms.head())
    print("222")
    thisdata.normalize_coords()
    print(thisdata.atoms.head())
    print("222")
    thisdata.zero_coords(thres=[0.5, 0.5, 0.5])
    INDs = thisdata.atoms.index.to_numpy()
    thisdata.remove_molecular_id()
    return thisdata, INDs


nsuper_vasp = 8
atom_style = "atomic"
#ref_data = lmpData.from_file("dumbbell011.dat", atom_style) 
#Use_INDs_ref = False
Use_SP = False
for ipot in range(len(Pots)):
#for ipot in range(1):
    pot = Pots[ipot]
    print("starting " + pot)
    label = Labels[ipot]
    thislatt = 2.85531162 #get_lattice(label, pot)
    for ipd in range(len(PDs)):
        pdt = PDs[ipd]
        nsuper = nsupers[ipd]
        gdata, avdata, spdata, idsp = get_data_DataOut(pot, pdt)
        if idsps is None:
            pass
        else:
            idsp = idsps[ipd + ipot*len(PDs)]
        print(f"natoms:{gdata.natoms} idsp:{idsp}")
        print("====")
        df_coords = get_df_coords(pot, pdt, idsp)
        zerodata = generate_empty_avdata(avdata)
        print(len(df_coords), len(df_coords[0]))
        data0 = df2data(zerodata, df_coords[0])
        print(data0.atoms.head())
        print("0000")
        x = data0.atoms['x'].to_numpy()
        y = data0.atoms['y'].to_numpy()
        z = data0.atoms['z'].to_numpy()
        cent_coords = np.array([np.mean(x), np.mean(y), np.mean(z)])

        print(f"centers:{cent_coords}")
        #for iref in range(len(a_refs)):
        for iref in range(1, len(a_refs)):
            ispin = iref + 1
            a_ref =  a_refs[iref]

            for isp in range(len(df_coords)):
            #for isp in range(2):
                print(f"isp:{isp} ispin:{ispin}")
                indata = df2data(zerodata, df_coords[isp])
                rmin1 = distance_check(indata)
                if isp == 0:
                    outdata, INDs = data2vasp(indata, cent_coords, nsuper, thislatt,
                                              a0=a_ref, INDs=None, nsuper_vasp=nsuper_vasp)
                    outdata0 = outdata.deepcopy()
                else:
                    outdata, IIs = data2vasp(indata, cent_coords, nsuper, thislatt,
                                             a0=a_ref, INDs=INDs, nsuper_vasp=nsuper_vasp)
                outtypes = outdata.atoms['type'].to_numpy()
                outtype2 = np.where(outtypes==2)
                displmax=get_displmax(outdata, outdata0)
                rmin2 = distance_check(outdata)
                print(f"isp:{isp} ispin:{ispin} natoms:{outdata.natoms}")
                print(f"rmin1:{rmin1} rmin2:{rmin2} dismax:{displmax}")
                print(f"type2:{outtype2} ")
                print("-------")
                '''
                fout = label + "_" + foldhead + "/"
                fout += label + "_" + pdt
                fout += "_ISPIN" + str(ispin) + "_" + str(isp) + ".POSCAR"
                outdata.to_POSCAR(fout)
                '''


                fout = label + "_" + foldhead + "/"
                fout += pdt + "_ISPIN" + str(ispin) + "_" + str(isp) + ".dat"
                indata.to_file(fout)

            print(f"finished: ispin:{ispin} ")
        print(f"finished: pd:{pdt} ")
    print("finished " + pot)
