import os

import pandas as pd
from pymatgen.io.lammps.outputs import parse_lammps_dumps, parse_lammps_log

from mylammps.inputs.data import lmpData
from mylammps.myglobal import config_vars

Work_Path = config_vars["Work_Path"]
outdata = "out.dat"
indata = "in.dat"


class ParseOuts:
    def __init__(self, foldheader, work_directory=Work_Path):
        self.work_directory = work_directory
        self.full_path = os.path.join(work_directory, foldheader)

    def parse_lmp_dumps(self, filepattern):
        thispath = os.getcwd()
        os.chdir(self.full_path)
        lammpsdumps = parse_lammps_dumps(filepattern)
        os.chdir(thispath)
        return lammpsdumps

    def parse_lmp_log(self, filename="log.lammps"):
        thispath = os.getcwd()
        os.chdir(self.full_path)
        runs = parse_lammps_log(filename=filename)
        os.chdir(thispath)
        return runs

    def parse_lmp_outdata(self, atom_style):
        thispath = os.getcwd()
        os.chdir(self.full_path)
        if os.path.isfile(outdata):
            outlmp = lmpData.from_file(outdata, atom_style=atom_style)
        elif os.path.isfile(indata):
            outlmp = lmpData.from_file(indata, atom_style=atom_style)
            #print("WARNING: Cannot find out.dat. Code parses in.dat.")
        else:
            raise ValueError("No Lammps data file.")
        os.chdir(thispath)
        return outlmp

    def parse_seakmc_summary(self):
        thispath = os.getcwd()
        os.chdir(self.full_path)
        df_seakmc_summary = pd.read_csv("Seakmc_summary.csv")
        os.chdir(thispath)
        return df_seakmc_summary

    def parse_seakmc_spout(self):
        thispath = os.getcwd()
        os.chdir(os.path.join(self.full_path, "SPOut"))
        df_seakmc_SPs = pd.read_csv("KMC_0_SPs.csv")
        os.chdir(thispath)
        return df_seakmc_SPs
