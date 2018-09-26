#!/usr/bin/env python2
# create_thumbnail.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-21-2018
# Updated Date: 09-21-2018

"""

create_thumbnail is used to generate image thumbnails for sour images for web browsing. Resized images will be save into a source resized bucket


 Temp:
 TODO:

 1.
 2.


    Current default S3 Bucket: s3://insight-data-images/Entity

    Run with .....:

    example:
            python producer.py --src_bucket_name "insight-data-images" --src_prefix "Entity/food/packaged_food/protein_bar/samples/" --des_bucket_name "insight-deep-images-hub"  --label_name "Think_thin_high_protein_caramel_fudge" --lon -73.935242 --lat 40.730610 --batch_id 1 --user_id 1


"""
from __future__ import print_function
import boto3
import os
import sys
import uuid
from PIL import Image
import PIL.Image

s3_client = boto3.client('s3')


def resize_image(image_path, resized_path):
    with Image.open(image_path) as image:
        image.thumbnail(tuple(x / 2 for x in image.size))
        image.save(resized_path)


def handler(event, context):
    for record in event['Records']:
        bucket = record['s3']['bucket']['name']
        key = record['s3']['object']['key']
        download_path = '/tmp/{}{}'.format(uuid.uuid4(), key)
        upload_path = '/tmp/resized-{}'.format(key)

        s3_client.download_file(bucket, key, download_path)
        resize_image(download_path, upload_path)
        s3_client.upload_file(upload_path, '{}resized'.format(bucket), key)