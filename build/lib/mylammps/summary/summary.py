import datetime
import os
import numpy as np
import pandas as pd

from mylammps.myglobal import config_vars, Constants
from mylammps.outputs.outputs import ParseOuts

Work_Path = config_vars["Work_Path"]
summary_cols = ["directory", "natoms", "volume", "Step", "Temp", "potential_energy", "kinetic_energy", "total_energy",
                "pressure", "epa", "epa_total_energy", "forward_barrier", "backward_barrier", "nSPs", "last_update"]


class mySummary:
    def __init__(self, work_directory=Work_Path, filename="mySummary.csv"):
        self.work_directory = work_directory
        self.filename = filename
        self.full_fname = os.path.join(work_directory, filename)

        if not os.path.isfile(self.full_fname):
            self.df = pd.DataFrame(columns=summary_cols)
            self.df.to_csv(self.full_fname, index=False, float_format=Constants["float_format"], sep="|")
        self.df = pd.read_csv(self.full_fname, sep="|")
        self.df = self.df.sort_values(summary_cols[0])
        self.df.sort_index(inplace=True)
        self.nentries = len(self.df)
        self.last_access = datetime.datetime.now()

    def to_file(self):
        self.df.to_csv(self.full_fname, index=False, float_format=Constants["float_format"], sep="|")

    @staticmethod
    def generate_default_entry(foldheader):
        thisdict = {}
        for key in summary_cols:
            if key == summary_cols[0]:
                thisdict[key] = foldheader
            elif key == summary_cols[-1]:
                thisdict[key] = str(datetime.datetime.now())
            else:
                thisdict[key] = 0.0
        return thisdict

    def delete_entries_by_inds(self, inds):
        if isinstance(inds, int) or isinstance(inds, np.int_):
            inds = [inds]
        for i in sorted(inds, reverse=True):
            self.df = self.df.drop(index=i)
        self.nentries = len(self.df)

    def select_by_directory(self, instring, style="INC"):
        inds = self.df.index.to_numpy()
        ys = self.df[summary_cols[0]].to_numpy()
        goods = []
        for i in range(len(ys)):
            f = ys[i]
            if style[0:3].upper() == "INC":
                if instring in f:
                    goods.append(inds[i])
            elif style[0:3].upper() == "EXA":
                if instring == f:
                    goods.append(inds[i])
            else:
                if instring not in f:
                    goods.append(inds[i])
        return np.array(goods)

    def select_by(self, by, condition):
        inds = self.df.index.to_numpy()
        ys = self.df[by].to_numpy()
        goods = np.compress(eval(condition), inds)
        return goods

    def get_an_entry(self, foldheader, atom_style, logfile="log.lammps", isSEAKMC=False):
        thisdict = mySummary.generate_default_entry(foldheader)
        parseout = ParseOuts(foldheader, work_directory=self.work_directory)
        outdata = parseout.parse_lmp_outdata(atom_style)
        volume = outdata.box.volume
        thisdict["natoms"] = outdata.natoms
        if not isSEAKMC:
            outruns = parseout.parse_lmp_log(filename=logfile)
            lastdf = outruns[-1]
            thisdict["Step"] = lastdf.at[len(lastdf) - 1, "Step"]
            thisdict["Temp"] = lastdf.at[len(lastdf) - 1, "Temp"]
            thisdict["potential_energy"] = lastdf.at[len(lastdf) - 1, "PotEng"]
            thisdict["kinetic_energy"] = lastdf.at[len(lastdf) - 1, "KinEng"]
            thisdict["total_energy"] = lastdf.at[len(lastdf) - 1, "TotEng"]
            thisdict["pressure"] = lastdf.at[len(lastdf) - 1, "Press"]
            thisdict["forward_barrier"] = 0.0
            thisdict["backward_barrier"] = 0.0
            thisdict["nSPs"] = 0
        else:
            lastdf = parseout.parse_seakmc_summary()
            thisdict["Step"] = lastdf.at[len(lastdf) - 1, "istep"]
            thisdict["Temp"] = 800.0
            thisdict["potential_energy"] = lastdf.at[len(lastdf) - 1, "ground_energy"]
            thisdict["kinetic_energy"] = 0.0
            thisdict["total_energy"] = lastdf.at[len(lastdf) - 1, "ground_energy"]
            thisdict["pressure"] = 0.0
            thisdict["forward_barrier"] = lastdf.at[len(lastdf) - 1, "forward_barrier"]
            thisdict["backward_barrier"] = lastdf.at[len(lastdf) - 1, "backward_barrier"]
            df_sps = parseout.parse_seakmc_spout()
            df_sps = df_sps.sort_values(["barrier"])
            barriers = df_sps["barrier"].to_numpy()
            if len(barriers) < 2:
                thisdict["nSPs"] = len(barriers)
            else:
                n = 1
                b0 = barriers[0]
                for i in range(1, len(barriers)):
                    vdiff = barriers[i]  - b0
                    if abs(vdiff) < 0.001:
                        n += 1
                    else:
                        break
                thisdict["nSPs"] = n

        thisdict["epa"] = thisdict["potential_energy"] / thisdict["natoms"]
        thisdict["epa_total_energy"] = thisdict["total_energy"] / thisdict["natoms"]
        thisdict["volume"] = volume

        return thisdict

    def update_an_entry(self, foldheader, thisdict):
        flist = self.df[summary_cols[0]].tolist()
        if foldheader in flist:
            ind = flist.index(foldheader)
            self.df.loc[ind] = thisdict
        else:
            self.df.loc[len(self.df)] = thisdict
        self.df = self.df.sort_values(summary_cols[0])
        self.nentries = len(self.df)

    def add_edit_an_entry(self, foldheader, atom_style, logfile="log.lammps", isSEAKMC=False):
        thisdict = self.get_an_entry(foldheader, atom_style, logfile=logfile, isSEAKMC=isSEAKMC)
        flist = self.df[summary_cols[0]].tolist()
        if foldheader in flist:
            ind = flist.index(foldheader)
            self.df.loc[ind] = thisdict
        else:
            self.df.loc[len(self.df)] = thisdict
        self.df = self.df.sort_values(summary_cols[0])
        self.nentries = len(self.df)
