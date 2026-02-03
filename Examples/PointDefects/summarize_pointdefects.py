import os,sys
import shutil
import copy
import numpy as np
from mylammps.inputs.data import lmpBox, lmpData
from mylammps.inputs.input import lmpInputs, SeakmcInputs
from mylammps.submission.submission import mySubmission
from mylammps.summary.summary import mySummary
from mylammps.elastic.elastic_constant import Elastic_properties
from mylammps.myglobal import config_vars

Input_Path = config_vars["Input_Path"]
Work_Path = config_vars["Work_Path"]

POTs = ["pot2.mtp", "pot3.mtp", "pot4.mtp"]
LattParas = [2.81951772, 2.8325172941, 2.836715819, 2.85, 2.85]
EPAs = [-5.1582565, -5.156386, -5.1555725, -8.4, -8.4]

thisSummary = mySummary(work_directory=Work_Path, filename="mySummary.csv")


fheaders = ["dumbbell011", "vacancy"]
atom_style = "atomic"
finput4lammps = "input.relax"

for i in range(2, len(POTs)):
    pot = POTs[i]
    fpotname = pot.replace(".mtp", "")
    a = LattParas[i]
    outfhead = os.path.join("PointDefects", fpotname)
    for k in range(len(fheaders)):
        fhead = fheaders[k]
        outfold = os.path.join(outfhead, fhead)
        outdict = thisSummary.get_an_entry(outfold, atom_style, logfile="log.lammps")
        ef = outdict['total_energy'] - outdict['natoms'] * EPAs[i]
        print(f"pot:{pot} fhead: {fhead} ef:{ef} epa:{outdict['epa']} epa_ref:{EPAs[i]}")
        thisSummary.update_an_entry(outfold, outdict)

thisSummary.to_file()
