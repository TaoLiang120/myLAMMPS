import os,sys
import shutil
import copy

from mylammps.inputs.data import lmpBox, lmpData

fname = "bcc110_112_111_orth4screw.data"
atom_style = "atomic"
force_field = ["Fe"]

a = 2.82814267976
lvac = 7.0*a

thisdata = lmpData.from_file(fname, atom_style, sort_id=False, parse_velocity=False)
thisdata.assert_force_field(force_field)
thisdata.scale_data(a, style=0)
supercell = [11, 7, 31]
outdata=thisdata.make_supercell(supercell)

burgerm = np.sqrt(3)*a/2.0
orientation = False
add_vacuum = False
direction = 0
outdata.create_screw_dislocation(burgerm, 1, style="bcc", handle_pbc="tilt",
                                 orientation=orientation,
                                 add_vacuum=add_vacuum, direction=direction, lvac=lvac)

outdata.to_file("screw2.dat")


## relax screw2.dat
## assume the relaxed screw2.dat is relaxed_screw2.dat 
thisdata = lmpData.from_file("relaxed_screw2.dat", atom_style)
direction = 0
lvac = 20
thisdata.add_vacuum(direction=direction, lvac=lvac, thres=[0.03*a, 0.03*a, 0.03*a])
newaxis = [2, 1, 0]
thisdata.swap_axes(newaxis)
thisdata.to_file("screw1.dat")

