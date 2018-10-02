# -*- coding: utf-8 -*-
#!/usr/bin/env python2
# .py
# ---------------
# Author: Zhongheng Li
# Init Date: 10-01-2018
# Updated Date: 10-01-2018

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
from datetime import date, datetime
import random
# from geojson import Point
import geojson
import pandas as pd
import logging
import os
from os.path import dirname as up





"""
Commonly Shared Statics

"""

# Set up project path
projectPath = up(up(os.getcwd()))


database_ini_file_path = "/utilities/database/database.ini"



logger = logging.getLogger()
logger.setLevel(logging.INFO)


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

retrieve batches info from DB

Temp workflow:

Select 100 most recent image batches are in New York


"""



def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""

    if isinstance(obj, (datetime,date)):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

def data2geojson(df):
    features = []
    insert_features = lambda X: features.append(
            geojson.Feature(geometry=geojson.Point((X["lon"],
                                                    X["lat"])),
                            properties=dict(neighbourhood=X["neighbourhood"],
                                            on_board_date=X["on_board_date"],
                                            batch_id=X["batch_id"],
                                            submitted_count=X["submitted_count"])))
    df.apply(insert_features, axis=1)
    with open('map1.geojson', 'w', encoding='utf8') as fp:
        geojson.dump(geojson.FeatureCollection(features), fp, sort_keys=True, ensure_ascii=False,  default=json_serial)



def get_image_batches_by_city(city):


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

        sql = "	SELECT ib.batch_id, ib.submitted_count, ib.on_board_date, pl.city, pl.neighbourhood, pl.lon, pl.lat \
                FROM images_batches AS ib " \
                "JOIN places as pl ON ib.place_id = pl.place_id " \
                "WHERE pl.city = %s " \
                "ORDER BY ib.on_board_date  DESC LIMIT 100;"


        # execute a statement
        print('Getting image urls for requesting labels ...')
        cur.execute(sql,(city,))

        results = cur.fetchall()


        # close the communication with the PostgreSQL
        cur.close()


        results_df = pd.DataFrame(results,
                                  columns=['batch_id', 'submitted_count',
                                           'on_board_date', 'city',
                                           'neighbourhood',
                                           'lon', 'lat'])

        print(results_df)

        results_geojson = data2geojson(results_df)

        print(results_geojson)

        # All labels ready return True
        return results_geojson

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')





city = 'NYC'

get_image_batches_by_city(city)