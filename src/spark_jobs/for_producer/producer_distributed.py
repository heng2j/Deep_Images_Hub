# -*- coding: utf-8 -*-
#!/usr/bin/env python2
# producer_distributed.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 10-09-2018

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

from argparse import ArgumentParser
from configparser import ConfigParser
import boto3
from io import BytesIO
import psycopg2
from psycopg2 import extras
from geopy.geocoders import Nominatim
import datetime
import random
from os.path import dirname as up



import logging
import time
import os
import io
import numpy as np

from os.path import dirname as up

import PIL
from PIL import Image


import pyspark.ml.linalg as spla
import pyspark.sql.types as sptyp
import numpy as np

from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.sql.types import StructType, StructField, IntegerType,StringType,BooleanType, Row





"""
Commonly Shared Statics

"""


sc = SparkContext(conf=SparkConf().setAppName("Build initial Imges dataset"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1

sqlContext = SQLContext(sc)


# Set up project path
projectPath = up(up(up(os.getcwd())))


database_ini_file_path = "/utilities/database/database.ini"

print("projectPath+database_ini_file_path: ", projectPath + database_ini_file_path)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Temp Lon and Lat list from NYC and San Francisco to simulate user locations
lon_list = [-73.935242,-74.005974,-73.989879,-73.984058,-122.419454,-122.470988,-122.448507]

lat_list = [40.730610,40.712776,40.734504,40.693165,37.780579,37.758063,37.791922]





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
Create Batch ID to keep track of this submission once the images are uploaded into Deep Image Hub


"""

def generate_new_batch_id(user_id,place_id,image_counter):


    sql = "INSERT INTO images_batches (user_id, ready, place_id, submitted_count, on_board_date ) VALUES %s RETURNING batch_id;"


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

        values_list = []

        values = (
                      user_id,
                      False,
                      place_id,
                      image_counter,
                      datetime.datetime.now()
                      )

        values_list.append(values)

        # writing image info into the database
        # execute a statement
        print('writing image batch info into the database...')
        psycopg2.extras.execute_values(cur, sql, values_list)
        # commit the changes to the database
        conn.commit()

        batch_id = cur.fetchone()[0]
        # close the communication with the PostgreSQL
        cur.close()

        return batch_id
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')




"""
Analysing Image Label

"""

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
        sql = "SELECT count(label_name)  FROM labels WHERE label_name = %s ;"


        # verify if label exist in the database
        # execute a statement
        print('Verifying if the label existed in the database...')
        cur.execute(sql,(label_name,))

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


# Recursively getting the parents' labels using Common Table Expressions(CTEs)
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
                  WHERE label_name = %s \
                  UNION ALL \
                  SELECT l.parent_name \
                  FROM labels l \
                  INNER JOIN labeltree ltree ON ltree.parent_name = l.label_name \
                      WHERE l.parent_name IS NOT NULL \
                ) \
                SELECT * \
                FROM labeltree;"



        # recursively split out the parent's label one by one to construct the path for the bucket's prefix
        # execute a statement
        print('Recursively getting the labels\' parents...')
        cur.execute(sql,(label_name,))

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

    for label in parent_labels:
        prefix = prefix + "/" + label

    return prefix


"""
Analysing geoinfo

"""


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

def generate_random_geo_point(lon,lat):

    dec_lat = random.random() / 20
    dec_lon = random.random() / 20

    new_lon = lon + dec_lon
    new_lat = lat + dec_lat

    return new_lon , new_lat


def getGeoinfo(lon,lat):
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    lat_lon_str = str(lat) + ", " + str(lon)
    print('lat_lon_str: ', lat_lon_str)

    location = geolocator.reverse(lat_lon_str)

    try:
        location.raw['address']['neighbourhood']
        return location.raw['place_id'], location.raw['licence'], location.raw['address']['postcode'], \
               location.raw['address']['neighbourhood'], location.raw['address']['city'], location.raw['address'][
                   'country']

    except KeyError as e:
        print("Can not find this address from Nominatim")
        return 1, "UNKNOWN", 0 , "UNKNOWN", "UNKNOWN", "UNKNOWN"

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
                       (" + str(image_info['place_id']) + ", '" + image_info['geo_licence'] + "', " + str(
            image_info['postcode']) + \
              ", '" + image_info['neighbourhood'] + "', '" + image_info['city'] + \
              "', '" + image_info['country'] + "', " + str(image_info['lon']) + \
              ", " + str(image_info['lat']) + ", '" + str(image_info['geo_point']) +"', (SELECT NOW()) ) \
                       ON CONFLICT(place_id)\
                       DO NOTHING RETURNING place_id;"

        # Insert geoinfo into database if place_id is not already exist

        # execute a statement
        print('Inserting geoinfo into database if place_id is not already exist...')

        # print(values)

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


def import_images_from_source(bucket,new_bucket, prefix, destination_prefix, image_info,new_keys,new_thumbnail_keys,new_small_thumbnail_keys):

    image_counter = 0


    for obj in bucket.objects.filter(Prefix=prefix).all():

        if '.jpg' in obj.key:

            img = Image.open(BytesIO(obj.get()['Body'].read()))

            img = img.resize((299, 299), PIL.Image.ANTIALIAS)

            in_mem_file = io.BytesIO()
            img.save(in_mem_file, format="JPEG")

            # Create the smaller thumbnail images for web
            img_small = img.resize((100, 100), PIL.Image.ANTIALIAS)
            in_mem_file_small = io.BytesIO()
            img_small.save(in_mem_file_small, format="JPEG")


            # Temp - Copy the the file from source bucket to destination bucekt
            old_source = {'Bucket': 'insight-data-images',
                          'Key': obj.key}

            new_key = obj.key.replace(prefix, "data/images" + destination_prefix + "/" + image_info['final_label_name'] + "/")

            filename = new_key.split('/')[-1].split('.')[0]

            print("Put file in to: ", new_key)
            print("filename: ", filename)

            new_obj = new_bucket.Object(new_key)
            new_obj.copy(old_source)



            new_keys.append(new_key)


            # Create thumbnails for Web

            new_thumbnail_key = obj.key.replace(prefix,
                                      "data/images/thumbnail" + destination_prefix + "/" + image_info['final_label_name'] + "/")

            new_small_thumbnail_key = obj.key.replace(prefix,
                                                      "data/images/thumbnail_small" + destination_prefix + "/" +
                                                      image_info[
                                                          'final_label_name'] + "/")

            thumbnail_path = "https://s3.amazonaws.com/insight-deep-images-hub/" + new_thumbnail_key

            samll_thumbnail_path = "https://s3.amazonaws.com/insight-deep-images-hub/" + new_small_thumbnail_key

            new_thumbnail_keys.append(thumbnail_path)
            new_small_thumbnail_keys.append(samll_thumbnail_path)

            # Put thumbnails to S3
            s3 = boto3.client('s3')
            s3.put_object(Body=in_mem_file.getvalue(), Bucket=des_bucket_name, Key=new_thumbnail_key, ContentType='image/jpeg', ACL='public-read')

            s3.put_object(Body=in_mem_file_small.getvalue(), Bucket=des_bucket_name, Key=new_small_thumbnail_key,
                          ContentType='image/jpeg', ACL='public-read')

            # increase image_counter by 1
            image_counter+=1



    return image_counter




"""
Save metadata in DB

"""

def write_imageinfo_to_DB(obj_keys, thumbnail_keys,small_thumbnail_keys,image_info):


    sql_images_insert = """ INSERT INTO \
     images(image_object_key,image_thumbnail_object_key,image_thumbnail_small_object_key, bucket_name, full_hadoop_path, parent_labels, label_name, batch_id, submission_time, user_id, place_id, image_index, embeddings, verified)\
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


        # update label's count TODO - Future development, No need to update count at the moment Update when Verified image label
        print('Updating the image counts for the label: ', image_info['final_label_name'])

        sql_update_counts_on_label = "UPDATE labels \
        SET image_count = image_count + %s \
        WHERE label_name = %s ; "

        values = (

            str(image_info['image_counter']),
            image_info['final_label_name']

        ,)



        cur.execute(sql_update_counts_on_label,values)


        # writing image info into the database
        # execute a statement
        print('writing images info into the database...')

        # create values list
        values_list = []

        # hadoop s3a prefix
        s3a_prefix = 's3a://'

        for i, obj_key in enumerate(obj_keys):

            values = (obj_key,
              thumbnail_keys[i],
              small_thumbnail_keys[i],
              image_info['destination_bucket'],
              s3a_prefix + image_info['destination_bucket'] + '/' + obj_key,
              image_info['destination_prefix'],
              image_info['final_label_name'],
              image_info['batch_id'],
              datetime.datetime.now(),
              image_info['user_id'],
              image_info['place_id'],
              None,
              None,
              True # TODO -- For now with out batch filtering
              )

            values_list.append(values)


        psycopg2.extras.execute_values(cur, sql_images_insert, values_list)
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



def create_spark_dataframe_from_list(label_list):
  # Create image categorical labels (integer IDs)
  local_rows = []
  for label in label_list:
    _row_struct = {"label": label }
    row = sptyp.Row(**_row_struct)
    local_rows.append(row)

  dataframe = sqlContext.createDataFrame(local_rows)
  return dataframe





def process_label(row):


    label_name = row.label.replace("\n","")


    random_index = random.randint(0,5)
    user_id = random.randint(1,5)

    lon = lon_list[random_index]
    lat = lat_list[random_index]

    # Set up geo points with geojson
    geo_point = (lon,lat)


    print("Current label: ", label_name)

    # # Variables
    # final_label_name = ""
    # parent_labels = []

    # Verifying Label if exist
    isLabel = verify_label(label_name)

    if isLabel == False:
        print("Sorry the supplying label doesn't exist in database")
        return None


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
                   "user_id"    : user_id,
                   "place_id"   : place_id,
                   "geo_licence"   : geo_licence,
                   "postcode"   : postcode,
                   "neighbourhood" : neighbourhood,
                   "city" : city,
                   "country" : country,
                   "geo_point" : geo_point,
                   "lon" : lon,
                   "lat" : lat

    }



    # Insert geoinfo into database if place_id is not already exist
    writeGeoinfo_into_DB(image_info)


    # Initiate an empty list of new object keys (as string) of where the image object locate at destinated S3 bucket
    new_keys = []

    # Initiate an empty list of new thumbnail keys (as string) of where the image object locate at destinated S3 bucket
    new_thumbnail_keys = []

    # Initiate an empty list of new small thumbnail keys (as string) of where the image object locate at destinated S3 bucket
    new_small_thumbnail_keys = []

    # From
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket(src_bucket_name)

    # To
    new_bucket = s3.Bucket(des_bucket_name)


    # Processing images
    image_counter = import_images_from_source(bucket,new_bucket, prefix + label_name + '/' , destination_prefix, image_info ,new_keys,new_thumbnail_keys,new_small_thumbnail_keys)

    print("Added "+ str(image_counter) + " images.")
    image_info['image_counter'] = image_counter


    batch_id = generate_new_batch_id(user_id, place_id,image_counter)

    print("batch_id:", batch_id)

    image_info['batch_id'] = batch_id


    # Bulk upload image info to database
    # write_imageinfo_to_DB(new_keys,images_features[0], image_info)
    write_imageinfo_to_DB(new_keys,new_thumbnail_keys,new_small_thumbnail_keys, image_info)

    return label_name, isLabel


if __name__ == '__main__':

    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-src_b", "--src_bucket_name", help="Source S3 bucket name", required=True)
    parser.add_argument("-src_p", "--src_prefix", help="Source S3 folder prefix", required=True)
    parser.add_argument("-src_t", "--src_type", help="From train, test or validation folder *Not needed for production for phone submission")
    parser.add_argument("-des_b", "--des_bucket_name", help="Destination S3 bucket name", required=True)
    parser.add_argument("-sf", "--source_file", help="source labels file path", required=True)


    args = parser.parse_args()

    # Assign input, output files and number of lines variables from command line arguments
    src_bucket_name = args.src_bucket_name
    des_bucket_name = args.des_bucket_name

    src_type = args.src_type


    prefix = args.src_prefix + src_type + '/'

    source_file_path = args.source_file

    label_list = []

    from pathlib import Path

    path = Path(source_file_path)

    with path.open() as f:
        label_list = list(f)

    print("label_list: ", label_list)
    labels_sdf = create_spark_dataframe_from_list(label_list)

    labels_sdf.show()


    labels_sdf = labels_sdf.repartition(10)

    schema = StructType([StructField("label", StringType()), StructField("isSubmitted", BooleanType(), False)])

    # Create result DF to check if all labels are processed
    result_df = (
        labels_sdf
            .rdd
            .map(process_label)
            .toDF(schema)
    )

    result_df.show()
