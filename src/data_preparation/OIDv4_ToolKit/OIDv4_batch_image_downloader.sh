#!/bin/bash
#
# Loop through the 600 classes file to download images from Open Images with OIDv4 ToolKit with batch size 10
# The script will automatically remove the labels that had been downloaded.
# Therefore if the download failed or took too much time it can be rerun.
#
# This script can be run as the following example:
#     sh OIDv4_batch_image_downloader.sh < Path to the 600_classes.txt file>
#
# Please run this script under a python3 env


FILE="$1"

while read -r one
do
  read -r two &&
  read -r three &&
  read -r four &&
  read -r five &&
  read -r six &&
  read -r seven &&
  read -r eight &&
  read -r nine &&
  read -r ten &&

  echo "Downloading images from the following labels:"
  printf "%s\n" "$one" "$two" "$three" "$four" "$five" "$six" "$seven" "$eight" "$nine" "$ten"
  ## or whatever you want to do to process those lines
  python3 main.py downloader --classes "$one" "$two" "$three" "$four" "$five" "$six" "$seven" "$eight" "$nine" "$ten" --type_csv train --image_IsOccluded 0 --noLabels --limit 2000
  echo END OF BATCH

done < ${FILE}
sed -i '1,11d' ${FILE}