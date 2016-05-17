#BSUB -n 2
#BSUB -q interactive
#BSUB -W 8:00



echo "Running on Images:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python -u test_net_on_set.py
