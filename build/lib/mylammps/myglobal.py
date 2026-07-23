import os
import subprocess
import sys

from monty.serialization import loadfn

try:
    config_vars = loadfn(os.path.join(os.path.expanduser('~'), 'myLAMMPS.yaml'))
except:
    sys.exit('No myLAMMPS.yaml file was found. Please configure the '
             ' myLAMMPS.yaml and put it in your home directory.')

Constants = {"kb": 8.6173324E-5, "bohr2angstrom": 0.529177249, "density2gcm": 1.6605402,
             "eVA2GPa": 160.2177, "GPa2eVA": 0.00624150648, "KJ2eVA": 1.0 / 96.4915666370759,
             "Tolerance": 0.1, "float_format": "%.8f"}

myElements = ["Fe", "H", "He", "Cr"]
myAtomicNumbers = [26, 1, 2, 24]
myAtomicVolumes = [11.3370564446, 0.5235987755982988, 11.0928837781, 11.4063765293]
myAtomicMasses = [55.845, 1.008, 4.003, 51.996]

def get_jobs_status(jobids=None):
    formats = 'jobid,jobname%10,workdir%120,elapsed%15,state%12,exitcode%10'
    if jobids is None:
        sacct_return = subprocess.Popen(
            "sacct -p --format " + formats, shell=True, stdout=subprocess.PIPE).stdout.readlines()
    else:
        if isinstance(jobids, dict):
            qjobs = jobids.keys()
        else:
            qjobs = jobids
        sacct_return = subprocess.Popen(
            'sacct -j %s -p --format %s' % (
                ','.join(qjobs), formats), shell=True, stdout=subprocess.PIPE).stdout.readlines()

    jobs_status = {}
    for el in sacct_return[1:]:
        keys = str(sacct_return[0])
        vals = str(el)
        d = dict(
            zip(keys.strip().split('|'), vals.strip().split('|')))
        if "b'JobID" in d:
            JobID = d["b'JobID"]
            JobID = JobID.replace("b'", "")
            d["JobID"] = JobID
            del d["b'JobID"]
        if '.' not in d['JobID']:
            jobs_status[d['JobID']] = d
    return jobs_status
