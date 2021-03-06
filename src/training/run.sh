#!/bin/bash


export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export PYSPARK_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export CORES_PER_WORKER=3
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export AWS_REGION=us-east-1

${SPARK_HOME}/bin/spark-submit \
--master ${MASTER} \
--packages databricks:spark-deep-learning:1.2.0-spark2.3-s_2.11 \
--conf spark.cores.max=${TOTAL_CORES} \
--conf spark.task.cpus=${CORES_PER_WORKER} \
--conf spark.task.maxFailures=1 \
--conf spark.stage.maxConsecutiveAttempts=1 \
--conf spark.executorEnv.JAVA_HOME="$JAVA_HOME" \
--conf spark.executorEnv.AWS_REGION=${AWS_REGION} \
--executor-memory 3g \
train_model_TF.py