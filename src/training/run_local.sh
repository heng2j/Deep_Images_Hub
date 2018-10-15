#!/bin/bash


cd ~/Deep_Images_Hub

python src/training/requester.py  --label_List Pen Table Lamp --user_id 2

model_id=$?


python src/training/cnn-keras/train.py --dataset /tmp/Deep_image_hub_Model_Training/dataset --model /tmp/Deep_image_hub_Model_Training/model/sample_model.model --plot /tmp/Deep_image_hub_Model_Training/model/plot.png --labelbin lb.pickle --training_request_number $model_id

python src/training/postTraining.py --training_request_number $model_id  --user_id 2