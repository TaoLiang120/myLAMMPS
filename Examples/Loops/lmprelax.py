from lammps import lammps
infile = "input.relax"
lmp = lammps()
lmp.file(infile)
lmp.close()
