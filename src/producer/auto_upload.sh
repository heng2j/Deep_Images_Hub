#!/bin/bash


export MASTER=spark://$(hostname):7077
export SPARK_WORKER_INSTANCES=3
export PYSPARK_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export PYSPARK_DRIVER_PYTHON=/home/ubuntu/shrink_venv/bin/python3
export CORES_PER_WORKER=1
export TOTAL_CORES=$((${CORES_PER_WORKER}*${SPARK_WORKER_INSTANCES}))
export AWS_REGION=us-east-1



loation_array[0]="--lon -74.005974 --lat 40.712776"
loation_array[1]="--lon -73.989879 --lat 40.734504"
loation_array[2]="--lon -73.984058 --lat 40.693165"
loation_array[3]="--lon -79.393225 --lat 43.660031"
loation_array[4]="--lon -122.419454 --lat 37.780579"
loation_array[5]="--lon -122.470988 --lat 37.758063"
loation_array[6]="--lon -122.448507 --lat  37.791922"
loation_array[7]="--lon -73.935242 --lat 40.730610"



LOCATIONS=("--lon -74.005974 --lat 40.712776" "--lon -73.989879 --lat 40.734504" "--lon -73.984058 --lat 40.693165" "--lon -79.393225 --lat 43.660031" "--lon -122.419454 --lat 37.780579" "--lon -122.470988 --lat 37.758063"  "--lon -122.448507 --lat  37.791922" "--lon -73.935242 --lat 40.730610")


# seed random generator
RANDOM=$$$(date +%s)

FILE=sample_labels.txt


while [ -s ${FILE} ]; do

    echo "file is not empty pop one label for simulating image submission"
    sleep 1 # throttle the check

    # pick a random entry from the locations list
    LON_LAT=${LOCATIONS[$RANDOM % ${#LOCATIONS[@]}]}

    echo ${LON_LAT}


    sed -e \$$'{w/dev/stdout\n;d}' -i~ ${FILE}



done
echo "file is empty - Stop sending images "
cat  s${FILE}









#time python producer_local.py --src_bucket_name "insight-data-images" --src_prefix "dataset/not_occluded/"  --src_type "test" --des_bucket_name "insight-deep-images-hub"  --label_name "Apple" ${LON_LAT} --user_id $(( ( RANDOM % 10 )  + 1 ))