#!/bin/bash


source ~/shrink_venv/bin/activate

LOCATIONS=(" --lon -74.005974 --lat 40.712776 " " --lon -73.989879 --lat 40.734504 " " --lon -73.984058 --lat 40.693165 " " --lon -122.413206 --lat 37.8006461 " " --lon -122.419454 --lat 37.780579 " " --lon -122.470988 --lat 37.758063 "  " --lon -122.448507 --lat  37.791922 " " --lon -73.935242 --lat 40.730610 ")


# seed random generator
RANDOM=$$$(date +%s)

FILE=sample_labels.txt


while [ -s ${FILE} ]; do

    echo "file is not empty pop one label for simulating image submission"
    sleep 1 # throttle the check

    # pick a random entry from the locations list
    LON_LAT=${LOCATIONS[$RANDOM % ${#LOCATIONS[@]}]}

    LABEL=$(echo | tail -1  ${FILE})

    time python producer_local.py --src_bucket_name "insight-data-images" --src_prefix "dataset/not_occluded/"  --src_type "test" --des_bucket_name "insight-deep-images-hub"  --label_name ${LABEL}  ${LON_LAT} --user_id $(( ( RANDOM % 10 )  + 1 ))

    echo "Done with the Label: "
    DONE_LABEL=$(echo | sed -e \$$'{w/dev/stdout\n;d}' -i~ ${FILE})
    echo ${DONE_LABEL}

done
echo "file is empty - Stop sending images "
cat  ${FILE}

