# -*- coding: utf-8 -*-
#!/usr/bin/env python2
# producer.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 09-21-2018

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
            python producer.py --src_bucket_name "insight-data-images" --src_prefix "Entity/food/packaged_food/protein_bar/samples/" --des_bucket_name "insight-deep-images-hub"  --label_name "Think_thin_high_protein_caramel_fudge" --lon -73.935242 --lat 40.730610 --batch_id 1 --user_id 1


"""
from __future__ import print_function
import sys
from argparse import ArgumentParser
from configparser import ConfigParser
import os
import boto3
from io import BytesIO
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import psycopg2
from psycopg2 import extras
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

# Set up project path
projectPath = up(up(os.getcwd()))

s3_bucket_name = "s3://insight-data-images/"

database_ini_file_path = "/utilities/database/database.ini"





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
Verify Batch ID

"""
## TODO - Maybe not in this Scope


"""
Analysing Image Label

1. Is the label existed (Created by user requests) ?
2. Pull the label info from DB 
2. Is the label belongs to any category or subcategory
    1. TODO: Check if the label belongs to any label within a category using classicfier to identify the object 
2. Is the label 

"""
## TODO


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

        # TODO Default raw values
        return location.raw['place_id'] , location.raw['licence'] , location.raw['address']['postcode'] , None ,location.raw['address']['city'],location.raw['address']['country']


    else:
        return location.raw['place_id'] , location.raw['licence'] , location.raw['address']['postcode'] , location.raw['address']['neighbourhood'],location.raw['address']['city'],location.raw['address']['country']


def writeGeoinfo_into_DB(image_info):
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
        sql = "INSERT \
                INTO \
                places(place_id, licence, postcode, neighbourhood, city, country, lon, lat, geometry, time_added) VALUES \
                (" + str(image_info['place_id']) + ", '"+ image_info['geo_licence'] + "', " + str(image_info['postcode']) + \
                ", '" + image_info['neighbourhood'] + "', '" + image_info['city'] + \
                "', '" + image_info['country'] + "', " + str(image_info['lon']) + \
                ", " + str(image_info['lat']) + ", NULL, (SELECT NOW()) ) \
                ON CONFLICT(place_id)\
                DO NOTHING RETURNING place_id;"






        # Insert geoinfo into database if place_id is not already exist

        # execute a statement
        print('Inserting geoinfo into database if place_id is not already exist...')
        print("sql: ", sql)
        cur.execute(sql)

        # commit the changes to the database
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')





"""
Fetch images, *compare image embeddings and put image to the proper folder in the AWS bucket

"""
## TODO



def processing_images(bucket,prefix,destination_prefix,image_info,new_keys_list):

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

            new_key = obj.key.replace(prefix, "data/images" + destination_prefix + "/" + image_info['final_label_name'] + "/")


            print("Put file in to: ", new_key)
            new_obj = new_bucket.Object(new_key)
            new_obj.copy(old_source)

            new_keys_list.append(new_key)


            # # TODO - decoupled this process to reduce DB access Save metadata in DB
            # write_imageinfo_to_DB(new_key, image_info)






"""
Save metadata in DB

"""
## TODO

def write_imageinfo_to_DB(obj_keys,image_info):


    sql = """ INSERT INTO \
     images(image_object_key, bucket_name, parent_labels, label_name, batch_id, submission_time, user_id, place_id, geometry, image_index, embeddings)\
     VALUES %s
     """


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

        # create values list
        values_list = []

        for obj_key in obj_keys:

            values = (obj_key,
                      image_info['destination_bucket'],
                      image_info['destination_prefix'],
                      image_info['final_label_name'],
                      image_info['batch_id'],
                      datetime.datetime.now(),
                      image_info['user_id'],
                      image_info['place_id'],
                      None,
                      None,
                      None
                      )

            values_list.append(values)

        # writing image info into the database
        # execute a statement
        print('writing images info into the database...')
        psycopg2.extras.execute_values(cur, sql, values_list)
        # commit the changes to the database
        conn.commit()

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



if __name__ == '__main__':

    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-src_b", "--src_bucket_name", help="Source S3 bucket name", required=True)
    parser.add_argument("-src_p", "--src_prefix", help="Source S3 folder prefix", required=True)
    parser.add_argument("-des_b", "--des_bucket_name", help="Destination S3 bucket name", required=True)
    parser.add_argument("-l", "--label_name", help="images label", required=True)
    parser.add_argument("-lon", "--lon", help="longitude", required=True)
    parser.add_argument("-lat", "--lat", help="latitude", required=True)
    parser.add_argument("-bid", "--batch_id", help="images batch id", required=True)
    parser.add_argument("-uid", "--user_id", help="supplier user id", required=True)



    args = parser.parse_args()

    # Assign input, output files and number of lines variables from command line arguments
    src_bucket_name = args.src_bucket_name
    des_bucket_name = args.des_bucket_name

    label_name = args.label_name
    lon = float(args.lon)
    lat = float(args.lat)
    batch_id = args.batch_id
    user_id = args.user_id
    prefix = args.src_prefix


    # From
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket(src_bucket_name)


    # To
    destination_prefix = ""

    new_bucket = s3.Bucket(des_bucket_name)

    # Temp Variables
    final_label_name = ""
    parent_labels = []
    batch_id = 1
    user_id = 1

    # Verifying Label if exist
    isLabel = verify_label(label_name)

    if isLabel == False:
        print("Sorry the supplying label doesn't exist in database")
        exit()


    final_label_name = label_name
    print("final_label_name: ", final_label_name)


    # Setting up the path for the prefix to save the images to the S3 bucket
    parent_labels = getParent_labels(label_name)
    destination_prefix = construct_bucket_prefix(parent_labels)


    # Analyzing geo info
    lon,lat = generate_random_geo_point(lon,lat)

    place_id, geo_licence, postcode, neighbourhood, city, country  =  getGeoinfo(lon,lat)

    image_info = { "destination_bucket" : des_bucket_name,
                   "destination_prefix" : destination_prefix,
                   "final_label_name" : final_label_name,
                   "batch_id"   : batch_id,
                   "user_id"    : user_id,
                   "place_id"   : place_id,
                   "geo_licence"   : geo_licence,
                   "postcode"   : postcode,
                   "neighbourhood" : neighbourhood,
                   "city" : city,
                   "country" : country,
                   "lon" : lon,
                   "lat" : lat

    }

    # Insert geoinfo into database if place_id is not already exist
    writeGeoinfo_into_DB(image_info)

    # Initiate an empty list of new object keys (as string) of where the image object locate at destinated S3 bucket
    new_keys = []

    # Processing images
    processing_images(bucket,prefix,destination_prefix,image_info,new_keys)

    # Bulk upload image info to database
    write_imageinfo_to_DB(new_keys, image_info)








