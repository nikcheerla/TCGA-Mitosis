
#BSUB -q gpu
#BSUB -n 3
#BSUB -W 1040:00
#BSUB -o BMMR-%J.out
#BSUB -e BMMR-%J.err
#BSUB -R "rusage[mem=60000, ngpus=1]"

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu0,gcc.cxxflags='-march=core2'" python -u BMMR-classify.py
echo "DONE BMMR:"
