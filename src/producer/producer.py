#!/usr/bin/env python2
# producer.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 09-19-2018

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
from configparser import ConfigParser
import os
import boto3
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import psycopg2
import json
import time
import datetime
from os.path import dirname as up


"""
Commonly Shared Statics

"""

# TODO - user env variable for username and password
# db_host = "insight-deep-images-hub.cj6pgqwql32a.us-east-1.rds.amazonaws.com"
# db_name = "insight_deep_images_hub"
# db_username = "heng"
# db_password = "deepimageshub"


# Set up project path
projectPath = up(up(os.getcwd()))

s3_bucket_name = "s3://insight-data-images/Entity"

database_ini_file_path = "/utilities/database/database.ini"




#
# bbox_labels_600_hierarchy_json_file = projectPath + "/data/labels/bbox_labels_600_hierarchy.json"

#
# print('bbox_labels_600_hierarchy_json_file_Path:', bbox_labels_600_hierarchy_json_file)


# entity_dict = json.loads(bbox_labels_600_hierarchy_json_file)



"""

config Database

"""

def config(filename=projectPath+database_ini_file_path, section='postgresql'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db


"""

Connect to DB


"""

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



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



label_name = ""
category_name = ""
subcategory_name = ""

#
# conn = psycopg2.connect( "host=" + db_name + " dbname=" + db_name + " user=" +  db_username + " password=" + db_password )

# TODO use the keyword arguments as the input
# conn = psycopg2.connect(host="localhost",database="suppliers", user="postgres", password="postgres")


"""
Analysing geoinfo

"""
## TODO


"""
Generating geoinfo


"""
## TODO

"""
Save metadata in DB

"""
## TODO




"""
Fetch images

"""
## TODO

s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket('insight-data-images')
prefix = "Entity/food/packaged_food/protein_bar/samples/"


# for obj in bucket.objects.filter(Prefix=prefix).all():
#
#     if '.jpg' in obj.key:
#
#         image = mpimg.imread(BytesIO(obj.get()['Body'].read()), 'jpg')
#
#         plt.figure(0)
#         plt.imshow(image)
#         plt.title('Sample Image from S3')
#         plt.pause(0.05)

"""
For each image

"""
## TODO


"""
Put image to the proper folder in the AWS bucket

"""
## TODO

# Boto 3
# s3.Object('mybucket', 'hello.txt').put(Body=open('/tmp/hello.txt', 'rb'))



if __name__ == '__main__':
    connect()


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



