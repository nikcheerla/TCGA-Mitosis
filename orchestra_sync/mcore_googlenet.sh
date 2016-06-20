
#BSUB -q mcore
#BSUB -n 3
#BSUB -W 10:00
#BSUB -o MCORETEST%J.out
#BSUB -e MCORETEST%J.err
#BSUB -R "rusage[mem=60000]"

echo "Training Network:"
python -u googlenet.py
