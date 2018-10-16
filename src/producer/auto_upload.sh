#!/bin/bash


# This script will take a file path for the source text file with a list of labels that want to submit the images with.
# This script will also take the subtype of the dataset either "test" "train" or "validation"
#
# Run this script with the following command:
#
#       sh ~/Deep_Images_Hub/src/producer/auto_upload.sh ~/Deep_Images_Hub/data/all_labels_For_Simulation.txt "validation"
#


source ~/shrink_venv/bin/activate

LOCATIONS=(" --lon -74.005974 --lat 40.712776 " " --lon -73.989879 --lat 40.734504 " " --lon -73.984058 --lat 40.693165 " " --lon -122.413206 --lat 37.8006461 " " --lon -122.419454 --lat 37.780579 " " --lon -122.470988 --lat 37.758063 "  " --lon -122.448507 --lat  37.791922 " " --lon -73.935242 --lat 40.730610 ")


# seed random generator
RANDOM=$$$(date +%s)

FILE="$1"
src_type="$2"


echo "file is not empty pop one label for simulating image submission"
sleep 1 # throttle the check

# pick a random entry from the locations list
LON_LAT=${LOCATIONS[$RANDOM % ${#LOCATIONS[@]}]}

LABEL=$(echo | tail -1  ${FILE})

time python ~/Deep_Images_Hub/src/producer/producer_local.py --src_bucket_name "insight-data-images" --src_prefix "dataset/not_occluded/"  --src_type ${src_type} --des_bucket_name "insight-deep-images-hub"  --label_name ${LABEL}  ${LON_LAT} --user_id $(( ( RANDOM % 10 )  + 1 ))



echo "Done with the Label: "
DONE_LABEL=$(echo | sed -e \$$'{w/dev/stdout\n;d}' -i~ ${FILE})
echo ${DONE_LABEL}



