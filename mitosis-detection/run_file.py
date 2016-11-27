import sys, os, subprocess


runfile = sys.argv[1]
gpu = sys.argv[2]
outfile = "debug/" + runfile[:-3] + "%J_prgmdata.out"
errfile = "debug/" + runfile[:-3] + "%J_prgmdata.err"
print sys.argv
if len(sys.argv) == 4:
    print subprocess.check_output(["bsub", "-q", gpu, "-n", sys.argv[3], "-W", "1:00", "-R", "rusage[mem=40000]", "-o", outfile,
        "-e", errfile, "python", "-u", runfile])
elif gpu == 'gpu':
    print subprocess.check_output(["bsub", "-q", "gpu", "-W", "140:00", "-R", "rusage[mem=30000, ngpus=1]", "-o", outfile,
        "-e", errfile,  "source /opt/gpu.sh; ", "THEANO_FLAGS=floatX=float32", "python", "-u", runfile])
else:
    print subprocess.check_output(["bsub", "-q", gpu, "-W", "140:00", "-R", "rusage[mem=40000]", "-o", outfile,
        "-e", errfile, "python", "-u", runfile])
