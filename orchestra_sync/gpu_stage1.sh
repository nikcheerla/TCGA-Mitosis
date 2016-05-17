
#BSUB -n 2
#BSUB -q gpu
#BSUB -W 24:00
#BSUB -o GPU%J.out
#BSUB -e GPU%J.err
#BSUB -R "rusage[mem=48000, ngpus=1]"

if ! [ -f "trainImg_stage1.npy" ]
  then
    echo "Preparing Data:"
    python data_stage1.py
fi

echo "Training Network:"
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python train_net_stage1.py
THEANO_FLAGS="floatX=float32,device=gpu1,gcc.cxxflags='-march=core2'" python data_stage2.py
