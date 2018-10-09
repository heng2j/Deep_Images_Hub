#!/bin/bash



split -l 160 ~/all_labels.txt ~/sample_labels_validation_

peg scp to-rem pySpark-cluster 1 ~/sample_labels_validation_aa /home/ubuntu/sample_labels_validation_aa

peg scp to-rem pySpark-cluster 2 ~/sample_labels_validation_ab /home/ubuntu/sample_labels_validation_ab

peg scp to-rem pySpark-cluster 3 ~/sample_labels_validation_ac /home/ubuntu/sample_labels_validation_ac

peg scp to-rem pySpark-cluster 4 ~/sample_labels_validation_ad /home/ubuntu/sample_labels_validation_ad


peg sshcmd-node pySpark-cluster 1 "nohup sh ~/Deep_Images_Hub/src/producer/auto_upload_for_batch.sh ~/sample_labels_validation_aa "validation"  > ~/Deep_Images_Hub/src/producer/auto_upload.log &" 

peg sshcmd-node pySpark-cluster 2 "nohup sh ~/Deep_Images_Hub/src/producer/auto_upload_for_batch.sh ~/sample_labels_validation_ab "validation"  > ~/Deep_Images_Hub/src/producer/auto_upload.log &" 

peg sshcmd-node pySpark-cluster 3 "nohup sh ~/Deep_Images_Hub/src/producer/auto_upload_for_batch.sh ~/sample_labels_validation_ac "validation"  > ~/Deep_Images_Hub/src/producer/auto_upload.log &" 

peg sshcmd-node pySpark-cluster 4 "nohup sh ~/Deep_Images_Hub/src/producer/auto_upload_for_batch.sh ~/sample_labels_validation_ad "validation"  > ~/Deep_Images_Hub/src/producer/auto_upload.log &" 




