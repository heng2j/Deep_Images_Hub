#!/bin/bash


export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export PYSPARK_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export AWS_REGION=us-east-1



. ./auto_upload.sh
