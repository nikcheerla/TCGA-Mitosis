# TCGA-Mitosis


Run the command bsub < gpu_stage1.sh on Orchestra to generate the data and train the first stage network.
Run the command bsub < gpu_stage2.sh on Orchestra to generate the data and train the second stage network.
Run the command test_net_on_set.py on Orchestra to parallelize and validate on all the pixels of the test images.

Alternatively, run the aptly named data_stage1.py, train_net_stage1.py, data_stage2.py, train_net_stage2.py to do the analysis locally.
