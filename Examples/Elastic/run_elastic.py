import os,sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData
from mylammps.inputs.input import lmpInputs, SeakmcInputs
from mylammps.submission.submission import mySubmission
from mylammps.summary.summary import mySummary
from mylammps.elastic.elastic_constant import Elastic_properties
from mylammps.myglobal import config_vars

cluster = "CAMM"
ncpu = 1
runtime = "720:00:00"
Execute = True

Input_Path = config_vars["Input_Path"]
Work_Path = config_vars["Work_Path"]

POTs = ["pot2.mtp", "pot3.mtp", "pot4.mtp"]
LattParas = [2.81951772, 2.8325172941, 2.836715819, 2.85, 2.85]  ## lattice parameters need to be updated 
fheaders = ["Fe2000"]
atom_style = "atomic"
finput4lammps = "input.relax"

distort_types = ["EOS", "OD", "MD"]
EOS_strains = [0.0, -0.01, -0.005, -0.002, 0.002, 0.005, 0.01]
OD_strains = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]
MD_strains = [0.0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01]
list_strains = [EOS_strains, OD_strains, MD_strains]

for i in range(len(POTs)):
    pot = POTs[i]
    fpotname = pot.replace(".mtp", "")
    a = LattParas[i]
    outfhead = os.path.join(Work_Path, "Elastic", fpotname)
    for k in range(len(fheaders)):
        fhead = fheaders[k]
        fname = os.path.join(Input_Path, "Elastic", fhead + ".dat")
        basedata = lmpData.from_file(fname, atom_style)
        basedata.scale_data(a, style=0)

        for idis in range(len(distort_types)):
            distype = distort_types[idis]
            strains = list_strains[idis]
            for istrain in range(len(strains)):
                strain = strains[istrain]
                outfold = os.path.join(outfhead, fhead, distype+str(istrain))
                if not os.path.isdir(outfold): os.makedirs(outfold)
                thisdata = basedata.deepcopy()
                thisdata.distort_data(strain, distype, dca=0, ca0=1.60, crystal="bcc")
                thisdata.to_file(os.path.join(outfold, "in.dat"))

                thisinput = lmpInputs(finput4lammps, fname4py_app="py_app.tmp")
                indict = {"timestep": 0.00025}
                thisinput.update_inputfile(**indict)

                mlipdict = {"outfname": None, "mtp-filename": pot}
                thisinput.update_external_file("mlip.ini", outfold, **mlipdict)

                copyfiles = [pot]
                thisinput.copy_files(copyfiles, outfold)

                finput = thisinput.to_file(outfold)
                fpyapp = thisinput.app_to_file(outfold, finput=finput)

                thissub = mySubmission(jobname="P"+str(i)+distype[0:1]+str(istrain), \
                                       cluster=cluster, ncpu=ncpu, runtime=runtime)
                thissub.to_file(outfold, fname_app=fpyapp)

                if Execute:
                    mySubmission.run(outfold)
