import os,sys
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

POTs = ["pot2.mtp", "pot3.mtp", "pot4.mtp"]
LattParas = [2.81951772, 2.8325172941, 2.836715819, 2.85, 2.85]
EPAs = [-5.1582565, -5.156386, -5.1555725, -8.4, -8.4]

fheaders = ["dumbbell011", "vacancy"]
atom_style = "atomic"
finput4seakmc = "seakmc_input.yaml"

cluster = "CAMM"
ncpu = 16
runtime = "720:00:00"
Execute = False
for i in range(2, len(POTs)):
    pot = POTs[i]
    a = LattParas[i]
    fpotname = pot.replace(".mtp", "")
    outfhead = os.path.join(Work_Path, "PointDefects", fpotname)
    for k in range(len(fheaders)):
        fhead = fheaders[k]
        fname = os.path.join(outfhead, fhead,  "out.dat")
        outfold = os.path.join(outfhead, fhead + "_seakmc")
        if not os.path.isdir(outfold): os.makedirs(outfold)
        thisdata = lmpData.from_file(fname, atom_style)
        thisdata.to_file(os.path.join(outfold, "in.dat"))

        thisinput = SeakmcInputs(finput4seakmc, fname4py_app="run_seakmc_p.py")
        indict = {"    NSearch": 20}
        thisinput.update_inputfile(**indict)

        mlipdict = {"outfname": None, "mtp-filename": pot}
        thisinput.update_external_file("mlip.ini", outfold, **mlipdict)

        copyfiles = [pot]
        thisinput.copy_files(copyfiles, outfold)

        finput = thisinput.to_file(outfold)
        fpyapp = thisinput.app_to_file(outfold)

        thissub = mySubmission(jobname=thisdata.full_formula, cluster=cluster, ncpu=ncpu, runtime=runtime)
        thissub.to_file(outfold, fname_app=fpyapp)

        if Execute:
            mySubmission.run(outfold)

