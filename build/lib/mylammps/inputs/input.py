# coding: utf-8
# Copyright (c) Tao Liang.
# Distributed under the terms of the MIT License.


__author__ = "Tao Liang"
__copyright__ = "Copyright 2021"
__version__ = "1.0"
__maintainer__ = "Tao Liang"
__email__ = "xhtliang120@gmail.com"
__date__ = "October 7th, 2021"

import os
import shutil

from mylammps.myglobal import config_vars

Input_Path = config_vars["Input_Path"]
Work_Path = config_vars["Work_Path"]


class lmpInputs:
    def __init__(
            self,
            finput,
            fname4py_app="py_app.tmp",
    ):
        self.finput = finput
        with open(Input_Path + "/" + finput, 'r') as f:
            self.lines = f.readlines()
        self.nlines = len(self.lines)

        fname = os.path.join(Input_Path + "/" + fname4py_app)
        with open(fname, 'r') as f:
            self.lines4py_app = f.readlines()
        self.nlines4py_app = len(self.lines4py_app)

    def update_inputfile(self, **kwargs):
        if kwargs:
            nlines = self.nlines
            for k in kwargs:
                v = kwargs[k]
                isreplaced = False
                for iline in range(nlines-1, -1, -1):
                    line = self.lines[iline]
                    if str(k) in line:
                        if not isreplaced:
                            self.lines[iline] = str(k) + " " + str(v) + "\n"
                            isreplaced = True
                if not isreplaced:
                    self.lines.append(str(k) + " " + str(v) + "\n")
                    self.nlines += 1

    def update_external_file(self, infname, outfold, outfname=None, **kwargs):
        with open(Input_Path + "/" + infname, 'r') as f:
            lines = f.readlines()
        nlines = len(lines)
        if kwargs:
            nlines_org = nlines
            for k in kwargs:
                v = kwargs[k]
                isreplaced = False
                for iline in range(nlines_org):
                    line = lines[iline]
                    if str(k) in line:
                        if not isreplaced:
                            lines[iline] = str(k) + " " + str(v) + "\n"
                            isreplaced = True
                if not isreplaced:
                    lines.append(str(k) + " " + str(v) + "\n")
                    nlines += 1

        if isinstance(outfname, str):
             pass
        else:
            outfname = infname


        with open(outfold + "/" + outfname, 'w') as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def copy_files(fnames, outfold):
        if isinstance(fnames, str):
            fnames = [fnames]
        for fname in fnames:
            shutil.copy(Input_Path + "/" + fname, outfold + "/" + fname)

    def to_file(self, outfold, finput="in.lammps"):
        with open(outfold + "/" + finput, 'w') as f:
            for line in self.lines:
                f.write(line)
        return finput

    def app_to_file(self, outfold, finput="in.lammps", outfname4py_app="application_py"):
        self.lines4py_app[1] = "infile = " + '"' + finput + '"' + "\n"
        with open(outfold + "/" + outfname4py_app, 'w') as f:
            for line in self.lines4py_app:
                f.write(line)
        return outfname4py_app

class SeakmcInputs:
    def __init__(
            self,
            finput,
            fname4py_app="run_seakmc_p.py",
    ):
        self.finput = finput
        with open(Input_Path + "/" + finput, 'r') as f:
            self.lines = f.readlines()
        self.nlines = len(self.lines)

        fname = os.path.join(Input_Path + "/" + fname4py_app)
        with open(fname, 'r') as f:
            self.lines4py_app = f.readlines()
        self.nlines4py_app = len(self.lines4py_app)

    def update_inputfile(self, **kwargs):
        if kwargs:
            nlines = self.nlines
            for k in kwargs:
                v = kwargs[k]
                for iline in range(nlines-1, -1, -1):
                    line = self.lines[iline]
                    if str(k) in line:
                        newlines = line.split(str(k))
                        newlines[-1] = str(k) + ": " + str(v) + "\n"
                        newline = ""
                        for j in range(len(newlines)):
                            newline += newlines[j]
                        self.lines[iline] = newline

    def update_external_file(self, infname, outfold, outfname=None, **kwargs):
        with open(Input_Path + "/" + infname, 'r') as f:
            lines = f.readlines()
        nlines = len(lines)
        if kwargs:
            nlines_org = nlines
            for k in kwargs:
                v = kwargs[k]
                isreplaced = False
                for iline in range(nlines_org):
                    line = lines[iline]
                    if str(k) in line:
                        if not isreplaced:
                            lines[iline] = str(k) + " " + str(v) + "\n"
                            isreplaced = True
                if not isreplaced:
                    lines.append(str(k) + " " + str(v) + "\n")
                    nlines += 1

        if isinstance(outfname, str):
             pass
        else:
            outfname = infname

        with open(outfold + "/" + outfname, 'w') as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def copy_files(fnames, outfold):
        if isinstance(fnames, str):
            fnames = [fnames]
        for fname in fnames:
            shutil.copy(Input_Path + "/" + fname, outfold + "/" + fname)

    def to_file(self, outfold):
        with open(outfold + "/" + "input.yaml", 'w') as f:
            for line in self.lines:
                f.write(line)

    def app_to_file(self, outfold, outfname4py_app="run_seakmc_p.py"):
        with open(outfold + "/" + outfname4py_app, 'w') as f:
            for line in self.lines4py_app:
                f.write(line)
        return outfname4py_app
