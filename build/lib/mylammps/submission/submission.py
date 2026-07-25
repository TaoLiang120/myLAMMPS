import os

from mylammps.myglobal import config_vars

VENV = config_vars["VENV"]


class mySubmission:
    def __init__(self, fname=None, jobname="Noname", cluster="CONDO", ncpu=48, runtime="720:00:00"):
        self.jobname = jobname
        cluster = cluster.upper()
        if not isinstance(fname, str):
            fname = os.path.join(os.path.dirname(__file__), cluster + ".JOB")
        with open(fname, 'r') as f:
            self.lines = f.readlines()
        if cluster == "CAMM":
            self.ncores = 224
            self.nthreads = 1
        elif cluster == "NERSC" or cluster == "NERSC_CPU":
            self.ncores = 128
            self.nthreads = 2
        elif cluster == "NERSC_GPU":
            self.ncores = 64
            self.nthreads = 2
        else:
            self.ncores = 48
            self.nthreads = 1

        if cluster == "OPEN":
            runtime = "24:00:00"

        self.ncpu = ncpu
        self.cluster = cluster
        self.runtime = runtime

        self.nnodes = self.get_nnodes(self.ncores)
        self.ntasks_per_node = int(self.ncpu / self.nnodes)

        if "NERSC" in self.cluster:
            if self.ntasks_per_node >= self.ncores:
                adj_ncores = int(self.ncores / 2)
                self.nnodes = self.get_nnodes(adj_ncores)
                self.ntasks_per_node = int(self.ncpu / self.nnodes)
        self.cpus_per_task = int(self.nthreads * self.ncores / self.ntasks_per_node)

        if self.ntasks_per_node > self.ncores:
            self.cpu_bind = "cores"
        else:
            self.cpu_bind = "thread"

    def get_nnodes(self, ncores):
        nnodes = int(self.ncpu / ncores)
        remainder = self.ncpu - nnodes * self.ncores
        if remainder > 0:
            nnodes += 1
        return nnodes

    def to_file(self, outfold, fname_app="application.py"):
        ##recommend to have a submission template for each cluster in global
        self.lines[1] = "#SBATCH -J " + self.jobname[0:6] + "\n"
        with open(outfold + "/submission.sh", 'w') as f:
            for line in self.lines:
                if line[0:15] == "#SBATCH --nodes":
                    f.write("#SBATCH --nodes=" + str(self.nnodes) + "\n")
                elif line[0:16] == "#SBATCH --ntasks":
                    f.write("#SBATCH --ntasks=" + str(self.ncpu) + "\n")
                elif line[0:14] == "#SBATCH --time":
                    f.write("#SBATCH --time=" + str(self.runtime) + "\n")
                elif line[0:25] == "#SBATCH --ntasks-per-node":
                    f.write("#SBATCH --ntasks-per-node=" + str(self.ntasks_per_node) + "\n")
                elif line[0:23] == "#SBATCH --cpus-per-task":
                    f.write("#SBATCH --cpus-per-task=" + str(self.cpus_per_task) + "\n")
                elif line[0:18] == "#SBATCH --cpu-bind":
                    f.write("#SBATCH --cpu-bind=" + str(self.cpu_bind) + "\n")
                elif line[0:14] == "conda activate":
                    f.write("conda activate " + VENV + "\n")
                elif line[0:15] == "srun --mpi=pmix":
                    f.write("srun --mpi=pmix -n " + str(self.ncpu) + " python " + fname_app + "\n")
                elif "srun -n" in line and " --cpu-bind " in line:
                    f.write("srun -n " + str(self.ncpu) + " --cpu-bind " + str(self.cpu_bind) + " python " + fname_app + "\n")
                elif line[0:7] == "srun -n":
                    f.write("srun -n " + str(self.ncpu) + " python " + fname_app + "\n")
                else:
                    f.write(line)

    @staticmethod
    def run(outfold):
        thispath = os.getcwd()
        os.chdir(outfold)
        os.system("sbatch submission.sh")
        os.chdir(thispath)


class mySubmissions(mySubmission):
    def __init__(self, root, fname=None, jobname="Noname", cluster="CONDO", ncpu=48, runtime="720:00:00"):
        super().__init__(fname=fname,
                         jobname=jobname,
                         cluster=cluster,
                         ncpu=ncpu,
                         runtime=runtime)
        self.root = root
        self.headers = self.lines[0:len(self.lines) - 1]
        self.batch_lines = []
        self.runline = self.lines[len(self.lines) - 1]
        self.get_batchjob_headers()

    def get_batchjob_headers(self):
        self.headers[1] = "#SBATCH -J " + self.jobname[0:6] + "\n"
        for i in range(len(self.headers)):
            line = self.headers[i]
            if line[0:15] == "#SBATCH --nodes":
                self.headers[i] = "#SBATCH --nodes=" + str(self.nnodes) + "\n"
            elif line[0:16] == "#SBATCH --ntasks":
                self.headers[i] = "#SBATCH --ntasks=" + str(self.ncpu) + "\n"
            elif line[0:14] == "#SBATCH --time":
                self.headers[i] = "#SBATCH --time=" + str(self.runtime) + "\n"
            elif line[0:25] == "#SBATCH --ntasks-per-node":
                self.headers[i] = "#SBATCH --ntasks-per-node=" + str(self.ntasks_per_node) + "\n"
            elif line[0:23] == "#SBATCH --cpus-per-task":
                self.headers[i] = "#SBATCH --cpus-per-task=" + str(self.cpus_per_task) + "\n"
            elif line[0:18] == "#SBATCH --cpu-bind":
                self.headers[i] = "#SBATCH --cpu-bind=" + str(self.cpu_bind) + "\n"
            elif line[0:14] == "conda activate":
                self.headers[i] = "conda activate " + VENV + "\n"

    def add_a_job(self, fold, fname_app="application.py"):
        thisstr = "cd " + fold + "\n"
        self.batch_lines.append(thisstr)
        if self.runline[0:15] == "srun --mpi=pmix":
            thisstr = "srun --mpi=pmix -n " + str(self.ncpu) + " python " + fname_app + "\n"
        elif "srun -n" in self.runline and " --cpu-bind " in self.runline:
            thisstr = "srun -n " + str(self.ncpu) + " --cpu-bind " + str(self.cpu_bind) + " python " + fname_app + "\n"
        elif self.runline[0:7] == "srun -n":
            thisstr = "srun -n " + str(self.ncpu) + " python " + fname_app + "\n"
        self.batch_lines.append(thisstr)
        self.batch_lines.append("\n")

    def to_batch(self):
        with open(self.root + "/submission.sh", 'w') as f:
            for line in self.headers:
                f.write(line)
            for line in self.batch_lines:
                f.write(line)