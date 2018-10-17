#!/bin/bash


# Upload images to S3 bucket
# Please modify the path accordingly

aws s3 cp --recursive /home/ubuntu/OIDv4_ToolKit/OID/Dataset/train s3://insight-data-images/dataset/not_occluded/train