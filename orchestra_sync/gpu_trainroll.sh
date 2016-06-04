
#BSUB -q gpu
#BSUB -W 104:00
#BSUB -o ROLL%J.out
#BSUB -e ROLL%J.err
#BSUB -R "rusage[mem=60000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python -u rolling_stack_training.py
