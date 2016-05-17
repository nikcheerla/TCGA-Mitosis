#BSUB -n 3
#BSUB -q gpu
#BSUB -W 240:00
#BSUB -o D2GPU%J.out
#BSUB -e D2GPU%J.err
#BSUB -R "rusage[mem=5000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu2,gcc.cxxflags='-march=core2'" python -u data_stage2.py
