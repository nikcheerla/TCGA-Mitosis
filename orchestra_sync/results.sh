#BSUB -n 4
#BSUB -q mcore
#BSUB -W 38:00
#BSUB -o RES%J.out
#BSUB -e RES%J.err
#BSUB -R "rusage[mem=10000]"

echo "Training Network:"
python -u f1score_result.py
