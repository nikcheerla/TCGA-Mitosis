
#BSUB -q gpu
#BSUB -n 3
#BSUB -W 1040:00
#BSUB -o 2G%J.out
#BSUB -e 2G%J.err
#BSUB -R "rusage[mem=60000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu0,gcc.cxxflags='-march=core2'" python -u googlenet.py
echo "DONE 2G:"
