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
from geopy.geocoders import Nominatim
import json
import time
import datetime
import random
import math
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

s3_bucket_name = "s3://insight-data-images/"

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
Variables

"""


final_label_name = ""
parent_labels = []


place_id = ""
geo_licence = ""
postcode = ""
neighbourhood = ""
city = ""
country = ""





image_path = ""
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


## Dummy Value - To be replaced with arguments
label_name = 'Think_thin_high_protein_caramel_fudge'

def verify_label(label_name):
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

        #TODO -  augmented SQL statement
        sql = "SELECT count(label_name)  FROM labels WHERE label_name = '" + label_name +"' ;"
        print("sql: ", sql)

        # verify if label exist in the database

        # execute a statement
        print('Verifying if the label existed in the database...')
        cur.execute(sql)

        result_count = cur.fetchone()[0]

        if result_count == 1:
            print("Label existed")
            return True
        else:
            print("Label doesn't exist")
            return False

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


# Recursively getting the parents' labels using Common Table Expressions (CTEs)
def getParent_labels(label_name):
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

        #TODO -  augmented SQL statement
        sql = "WITH RECURSIVE labeltree AS ( \
                SELECT parent_name \
                FROM labels \
                  WHERE label_name = '" + label_name + "' \
                  UNION ALL \
                  SELECT l.parent_name \
                  FROM labels l \
                  INNER JOIN labeltree ltree ON ltree.parent_name = l.label_name \
                      WHERE l.parent_name IS NOT NULL \
                ) \
                SELECT * \
                FROM labeltree;"


        print("sql: ", sql)

        # recursively split out the parent's label one by one to construct the path for the bucket's prefix
        # execute a statement
        print('Recursively getting the labels\' parents...')
        cur.execute(sql)

        row = cur.fetchone()

        parent_labels = []

        while row is not None:
            parent_labels.insert(0,row[0])
            row = cur.fetchone()

        return parent_labels


        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


# Construct the path for the bucket's prefix
def construct_bucket_prefix(parent_labels):

    prefix = ""

    # '/tmp/hello.txt'
    for label in parent_labels:
        prefix = prefix + "/" + label

    return prefix


"""
Analysing geoinfo

"""
## TODO


"""
Generating geoinfo

    # Raw Location Data Sample:
    # {'place_id': '138622978', 'licence': 'Data © OpenStreetMap contributors, ODbL 1.0. https://osm.org/copyright',
    #  'osm_type': 'way', 'osm_id': '265159179', 'lat': '40.7364439', 'lon': '-73.9339868252163',
    #  'display_name': '51-27, 35th Street, Blissville, Queens County, NYC, New York, 11101, USA',
    #  'address': {'house_number': '51-27', 'road': '35th Street', 'neighbourhood': 'Blissville',
    #              'county': 'Queens County', 'city': 'NYC', 'state': 'New York', 'postcode': '11101', 'country': 'USA',
    #              'country_code': 'us'}, 'boundingbox': ['40.7362729', '40.7365456', '-73.9340831', '-73.9338454']}

"""
## TODO

def generate_random_geo_point(lon,lat):

    dec_lat = random.random() / 20
    dec_lon = random.random() / 20

    new_lon = lon + dec_lon
    new_lat = lat + dec_lat

    return new_lon , new_lat


def getGeoinfo(lon,lat):
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    lon_lat_str = str(lat) + ", " + str(lon)
    print('lon_lat_str: ', lon_lat_str)

    location = geolocator.reverse(lon_lat_str)

    if location.raw['address']['neighbourhood'] == None:

        return location.raw['place_id'] , location.raw['licence'] , location.raw['address']['postcode'] , None ,location.raw['address']['city'],location.raw['address']['country']


    else:
        return location.raw['place_id'] , location.raw['licence'] , location.raw['address']['postcode'] , location.raw['address']['neighbourhood'],location.raw['address']['city'],location.raw['address']['country']




"""
Fetch images

"""
## TODO



# From
s3 = boto3.resource('s3', region_name='us-east-1')
bucket = s3.Bucket('insight-data-images')
prefix = "Entity/food/packaged_food/protein_bar/samples/"

# To
destination_bucket = "insight-deep-images-hub"
destination_prefix = ""


new_bucket = s3.Bucket(destination_bucket)

def processing_images(prefix,destination_prefix):

    for obj in bucket.objects.filter(Prefix=prefix).all():

        if '.jpg' in obj.key:

            # TODO - Processing Images
            image = mpimg.imread(BytesIO(obj.get()['Body'].read()), 'jpg')

            # plt.figure(0)
            # plt.imshow(image)
            # plt.title('Sample Image from S3')
            # plt.pause(0.05)

            # Temp - Copy the the file from source bucket to destination bucekt
            old_source = {'Bucket': 'insight-data-images',
                          'Key': obj.key}

            new_key = obj.key.replace(prefix, "data/images" + destination_prefix)
            print("Put file in to: ", new_key)
            new_obj = new_bucket.Object(new_key)
            new_obj.copy(old_source)





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



"""
Save metadata in DB

"""
## TODO




if __name__ == '__main__':

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


    # Verifying Label if exist
    isLabel = verify_label(label_name)

    if isLabel == False:
        print("Sorry the suppplying label doesn't exist in database")
        exit()


    final_label_name = label_name
    print("final_label_name: ", final_label_name)


    # Setting up the path for the prefix to save the images to the S3 bucket
    parent_labels = getParent_labels(label_name)
    print(parent_labels)

    destination_prefix = construct_bucket_prefix(parent_labels)

    print(destination_prefix)

    processing_images(prefix, destination_prefix)

    # Analyzing geo info
    ## Dummy Value - TODO replaced with arguments
    lon = -73.935242
    lat = 40.730610

    lon,lat = generate_random_geo_point(lon,lat)

    place_id, geo_licence, postcode, neighbourhood, city, country  =  getGeoinfo(lon,lat)


    print(place_id, geo_licence, postcode, neighbourhood, city, country)

    # Save Batch images info to database








