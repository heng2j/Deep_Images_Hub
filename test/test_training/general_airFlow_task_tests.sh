#!/bin/bash


# This script are testing the tasks within 2 dags with ids "image_submission_simulation" and  "local_model_training"
# This script can run simply with : sh general_airFlow_task_tests.sh
#


source ~/shrink_venv/bin/activate

cd ~/Deep_Images_Hub/src/airflow/workspace

export AIRFLOW_HOME=`pwd`/airflow_home
export AIRFLOW_GPL_UNIDECODE=yes

airflow version
airflow initdb


airflow webserver -p 8081 &
airflow scheduler &


cd ~

# Testing the tasks in dags

airflow test image_submission_simulation Distribute_labels 2017-07-01

airflow test image_submission_simulation Upload_images_From_Node_1 2017-07-01

airflow test image_submission_simulation Upload_images_From_Node_2 2017-07-01

airflow test local_model_training training_model_in_action 2017-07-01

airflow test local_model_training post_training_processing 2017-07-01