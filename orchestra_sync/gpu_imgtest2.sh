#BSUB -n 4
#BSUB -q mcore
#BSUB -W 38:00
#BSUB -o img_debug/IMG%J.out
#BSUB -e img_debug/IMG%J.err
#BSUB -R "rusage[mem=10000]"

echo "Training Network:"
python -u img_read.py test/A00_00.bmp
