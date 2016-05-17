#BSUB -n 2
#BSUB -q short
#BSUB -W 4:00
#BSUB -o CPU%J.out
#BSUB -e CPU%J.err
#BSUB -R "rusage[mem=8000]"

THEANO_FLAGS="floatX=float32" python data_stage1.py
