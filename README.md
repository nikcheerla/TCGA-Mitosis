# TCGA-Mitosis

<<<<<<< HEAD
The folder orchestra_sync contains the scripts used to run the analyses.

orchestra_sync/rolling_stack_training.py is the most recent, up to data one that trains a convolutional netwrok while using a boosting algorithm to improve results.

The folder orchestra_images contains some of the image results data.

The folder cloud_server contains an entire, fully contained web app for crowdsourcing the annotation of mitotic figures.

Folders that **must** be downloaded and are not provided in this repo include train, a folder that contained mitotic slide images and can be downloaded from the ludo17.free.fr/mitos_2012/download.html ICPR competition, and test, a folder that can be downloaded from the same place.


=======

Run the command bsub < gpu_stage1.sh on Orchestra to generate the data and train the first stage network.
Run the command bsub < gpu_stage2.sh on Orchestra to generate the data and train the second stage network.
Run the command test_net_on_set.py on Orchestra to parallelize and validate on all the pixels of the test images.

Alternatively, run the aptly named data_stage1.py, train_net_stage1.py, data_stage2.py, train_net_stage2.py to do the analysis locally.
>>>>>>> 4e68356f45cc7726bc5dd45bc08d66c7dfe75b65
