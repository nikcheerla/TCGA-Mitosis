
#BSUB -n 2
#BSUB -q gpu
#BSUB -W 24:00
#BSUB -o GPU%J.out
#BSUB -e GPU%J.err
#BSUB -R "rusage[mem=32000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python simple_net_example.py
