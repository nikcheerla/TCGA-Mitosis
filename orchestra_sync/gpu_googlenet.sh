
#BSUB -q gpu
#BSUB -W 140:00
#BSUB -o MULTI%J.out
#BSUB -e MULTI%J.err
#BSUB -R "rusage[mem=60000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python -u googlenet.py
echo "GOOGLE MULTI"
