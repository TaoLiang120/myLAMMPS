import os, sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData
from mylammps.inputs.input import lmpInputs, SeakmcInputs
from mylammps.submission.submission import mySubmission
from mylammps.summary.summary import mySummary
from mylammps.elastic.elastic_constant import Elastic_properties
from mylammps.myglobal import config_vars

Input_Path = config_vars["Input_Path"]
Work_Path = config_vars["Work_Path"]

POTs = ["pot1.mtp", "pot2.mtp", "pot3.mtp", "pot4.mtp", "pot5.mtp"]
LattParas = [2.85, 2.85, 2.85, 2.85, 2.85]

thisSummary = mySummary(work_directory=Work_Path, filename="mySummary.csv")

fheaders = ["Fe2000"]
atom_style = "atomic"
finput4lammps = "input.relaxbox"

for i in range(len(POTs)):
    pot = POTs[i]
    fpotname = pot.replace(".mtp", "")
    a = LattParas[i]
    outfhead = os.path.join("Grounds", fpotname)
    for k in range(len(fheaders)):
        fhead = fheaders[k]
        outfold = os.path.join(outfhead, fhead)
        thisSummary.add_edit_an_entry(outfold, atom_style, logfile="log.lammps")

thisSummary.to_file()
