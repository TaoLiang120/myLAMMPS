import os,sys
import shutil
import copy
import numpy as np
import pandas as pd

from mylammps.inputs.data import lmpBox, lmpData
from mylammps.inputs.input import lmpInputs, SeakmcInputs
from mylammps.submission.submission import mySubmission
from mylammps.summary.summary import mySummary
from mylammps.elastic.elastic_constant import Elastic_properties
from mylammps.myglobal import config_vars

Input_Path = config_vars["Input_Path"]
Work_Path = config_vars["Work_Path"]

atom_style = "atomic"
thisSummary = mySummary(work_directory=Work_Path, filename="mySummary.csv")

POTs = ["pot2.mtp", "pot3.mtp", "pot4.mtp"]
LattParas = [2.81951772, 2.8325172941, 2.836715819, 2.85, 2.85]
EPAs = [-8.4, -8.4, -8.4, -8.4, -8.4]

fheaders = ["Fe2000"]
finput4lammps = "input.relax"

distort_types = ["EOS", "OD", "MD"]
EOS_strains = [0.0, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01]
OD_strains = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]
MD_strains = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]
list_strains = [EOS_strains, OD_strains, MD_strains]
Elastic_cols = ["Pot", "data", "Bulk", "volume", "C11", "C12", "C44", "Youngs", "Shear", "Poisson"]
fElastic = "Elastic_summary.csv"
df_elastic = pd.DataFrame(columns=Elastic_cols)

def generate_default_dict(pot, data, cols):
    thisdict = {}
    for key in cols:
        if key == "Pot":
            thisdict[key] = pot
        elif key == "data":
            thisdict[key] = data
        else:
            thisdict[key] = 0.0
    return thisdict

def assign2dict(thisdict, indict):
    thisdict["Bulk"] = indict["B"]
    thisdict["volume"] = indict["v0"]
    thisdict["C11"] = indict["c11"]
    thisdict["C12"] = indict["c12"]
    thisdict["C44"] = indict["c44"]
    thisdict["Youngs"] = indict["Y"]
    thisdict["Shear"] = indict["G"]
    thisdict["Poisson"] = indict["v"]
    return thisdict

for i in range(len(POTs)):
    pot = POTs[i]
    fpotname = pot.replace(".mtp", "")
    a = LattParas[i]
    outfhead = os.path.join("Elastic", fpotname)

    for k in range(len(fheaders)):
        fhead = fheaders[k]
        thisdict = generate_default_dict(pot, fhead, Elastic_cols)

        for idis in range(len(distort_types)):
            distype = distort_types[idis]
            strains = list_strains[idis]
            volumes = []
            energies = []
            for istrain in range(len(strains)):
                strain = strains[istrain]
                outfold = os.path.join(outfhead, fhead, distype+str(istrain))

                outdict = thisSummary.get_an_entry(outfold, atom_style, logfile="log.lammps")
                volumes.append(outdict["volume"])
                energies.append(outdict["total_energy"])

                thisSummary.update_an_entry(outfold, outdict)

            if distype == "EOS":
                EOS_dict = Elastic_properties.EOS_fit(volumes, energies)
                print(EOS_dict)
            elif distype == "OD":
                OD_dict = Elastic_properties.distort_fit(strains, energies)
                print(OD_dict)
            else:
                MD_dict = Elastic_properties.distort_fit(strains, energies)
                print(MD_dict)

        outdict = Elastic_properties.get_cubic_elastic_constants(EOS_dict["B"], EOS_dict["v0"], OD_dict["c_fit"], MD_dict["c_fit"])
        thisdict = assign2dict(thisdict, outdict)
        df_elastic.loc[len(df_elastic)] = thisdict


thisSummary.to_file()
df_elastic.to_csv(fElastic, index=False, float_format="%.8f")
