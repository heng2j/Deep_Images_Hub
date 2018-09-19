#!/usr/bin/env python2
# producer.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 09-18-2018

"""

Producer is used to process incoming images that are sending as a batch from the user.
The producer will perform the following tasks to process the images:

 Temp: Take images from S3 bucket
 TODO: Accept images from user submissions from iOS devices

 1. Intake image
 2. Classify Label  - Temp with dictionary / TODO with WordNet
 3. Take Geoinfo - Temp with auto generated lat & lon / TODO with geographical info from image metadata
 4. Put the image into an existing folder with existing label. Temp - Create new folder if label is not existed.
 4. Insert image metadata into PostgreSQL database: image path on S3, label, category, subcategory, geometry, city, country, timestamp



    Current default S3 Bucket: s3://insight-data-images/Entity

    Run with .....:

    example:
            producer.py "s3://insight-data-images/Entity/food/packaged_food" "Clif_protein_bar_vanilla_almond"



"""
import sys
from argparse import ArgumentParser
import os
import boto3
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import json
import time
import datetime
from os.path import dirname as up


"""
Commonly Shared Statics

"""
# Set up project path
projectPath = up(up(os.getcwd()))

s3_bucket_name = "s3://insight-data-images/Entity"

bbox_labels_600_hierarchy_json_file = projectPath + "/data/labels/bbox_labels_600_hierarchy.json"


print('bbox_labels_600_hierarchy_json_file_Path:', bbox_labels_600_hierarchy_json_file)


# entity_dict = json.loads(bbox_labels_600_hierarchy_json_file)



"""
Variables

"""



image_path = ""
label_name = ""
category_name = ""
subcategory_name = ""
geo_dot = ""
city = ""
country = ""
timestamp = ""



"""
Analysing Image Label

1. Is the label existed (Created by user requests) ?
2. Pull the label info from DB 
2. Is the label belongs to any category or subcategory
    1. TODO: Check if the label belongs to any label within a category using classicfier to identify the object 
2. Is the label 

"""
## TODO


"""
Fetch images

"""
## TODO

temp_path = 'insight-data-images/Entity/food/packaged_food/protein_bar/Think_thin_high_protein_caramel_fudge/'


resource = boto3.resource('s3', region_name='us-east-1')
bucket = resource.Bucket('insight-data-images')



image_object = bucket.Object('Entity/food/packaged_food/protein_bar/Think_thin_high_protein_caramel_fudge/IMG_1673.jpg')

image = mpimg.imread(BytesIO(image_object.get()['Body'].read()), 'jpg')

plt.figure(0)
plt.imshow(image)
plt.title('Sample Image from S3')
plt.pause(2)


"""
For each image

"""
## TODO


"""
Put image to the proper folder in the AWS bucket

"""
## TODO


"""
Generate geoinfo

"""
## TODO



"""
Save metadata in DB

"""
## TODO



# if __name__ == '__main__':
#     # Set up argument parser
#     parser = ArgumentParser()
#     parser.add_argument("-b", "--bucketPath", help="S3 bucket path", required=True)
#     parser.add_argument("-l", "--labelName", help="images label", required=True)
#
#     args = parser.parse_args()
#
#     # Assign input, output files and number of lines variables from command line arguments
#     bucketPath = args.bucketPath
#     labelName = args.labelName



