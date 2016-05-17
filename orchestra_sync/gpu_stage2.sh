
#BSUB -q gpu
#BSUB -W 84:00
#BSUB -o 2GPU%J.out
#BSUB -e 2GPU%J.err
#BSUB -R "rusage[mem=40000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python -u train_net_stage2.py
