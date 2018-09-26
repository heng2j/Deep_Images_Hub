#!/usr/bin/env python2
# requester.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-21-2018
# Updated Date: 09-21-2018

"""

requester is used by end users to make request on image classicfication models with their choices of classes.

 Temp: ...
 TODO: ...

 Given: user_id, classes_list, destination_bucket

 1. Get user_info from user_id
 2. Verify classes
    1. Check the images counts for each given label is there enough images for training - Threshold: 500
 3. If there are enough images:
        1. Train the model with given labels of images
            1. Bring up the Training Cluster with EMR with specified AMI
                1. Lambda Function when ready trigger train
                2. When training is done, send the zip and send the model to CPU node to create the CoreML model
                    1. Start Tiering Down the GPU cluster
                3. When CoreML model is created, send both the TF trained model, Training Plot and CoreML model to the user's bucket
                4. Once completed tier down the last CPU node.
                5. Notify user by e-mail the model is ready with Lambda funciton

            2. Train the model
            3. Send the model back to
        2. Convert the model with to CoreML model
        3. Send both the weights and CoreML model to user's bucket
        4. Notify user when ready.
    If there is not enough images for training:
        1. Store the shorted labels into a list
        2. Send user an e-mail to notify him that there is not enough trainig data for the listing labels at the moment.
 **4. Use is able to subscribe to a label of images Subscribe to a



    Run with .....:

    example:
        requester.py "s3://insight-deep-images-hub/users/username_organization_id/models/packaged_food"

        python requester.py --des_bucket_name "insight-deep-images-hub" --des_prefix "user/userid/model/" --label_List 'Apple' 'Banana' 'protein_bar'  --user_id 2


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
Analysing Image Labels

Temp workflow:
1. Loop though the list of labels
    1. Is the label existed?
        1. If not, inform user and exit
    2. Is the label currently have enough images?
        1. If Yes, proceed
        2. If No
            1. Insert into label_watch_list table in the DB
            2. Inform user label not ready
                1. Either user can wait - show waiting message and constantly monitoring the jobs
                2. or exist and execute the command again  
    3. Return all labels ready Flag
2. If all Labels ready kick start the training model training process

"""
## TODO
def verify_labels(label_list):
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

        print('Verifying if the labels existed in the database...')
        for label_name in label_list:

            #TODO -  augmented SQL statement
            sql = "SELECT count(label_name)  FROM labels WHERE label_name = '" + label_name +"' ;"
            print("sql: ", sql)

            # verify if label exist in the database
            # execute a statement
            cur.execute(sql)

            result_count = cur.fetchone()[0]

            if result_count == 1:
                print("Label existed")

            else:
                print("Label '" + label_name +"' doesn't exist")
                return False

                # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return True

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

# Verify if the labels currently have enough images
def verify_labels_quantities(label_list,user_info):

    user_id = user_info['user_id']

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

        print('Verifying if the labels has enough images...')
        for label_name in label_list:

            #TODO -  augmented SQL statement
            sql = "SELECT label_name FROM labels WHERE label_name in ('Apple', 'Banana','protein_bar' ) AND image_count < 100;"

                  # " '" + label_name +"' ;"
            print("sql: ", sql)

            # verify if label exist in the database
            # execute a statement
            cur.execute(sql)

            results = cur.fetchall()

            # The returning results are the labels that doesn't have enough images
            if results:
                print("The following labels does not have enough images:")
                print (results)

                print("These labels will save into the requesting_label_watchlist table")

                # TODO - Save labels into the requesting_label_watchlist table
                save_to_requesting_label_watchlist(cur, results, user_id)

                # commit changes
                conn.commit()

                return False

            else:
                print("All labels are ready for training :) ")
                return True

                # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return True

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


# Save labels into the requesting_label_watchlist table
def save_to_requesting_label_watchlist(cur,label_list,user_id):


    print('Saving labels into the requesting_label_watchlist table...')
    for label_name in label_list:

        # TODO -  augmented SQL statement

        sql = " INSERT INTO requesting_label_watchlist (label_name, user_ids,last_requested_userid, new_requested_date ) VALUES \
        ( '"+ label_name[0] +"',ARRAY[" + user_id + "], " + user_id + ", (SELECT NOW()) ) ON CONFLICT (label_name)\
        DO UPDATE \
        SET user_ids = array_append(requesting_label_watchlist.user_ids, "+ user_id +"),\
        last_requested_userid = " + user_id + " \
        WHERE requesting_label_watchlist.label_name = '" + label_name[0] + "';"

        print("sql: ", sql)

        # verify if label exist in the database
        # execute a statement
        cur.execute(sql)






if __name__ == '__main__':

    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-des_b", "--des_bucket_name", help="Destination S3 bucket name", required=True)
    parser.add_argument("-des_p", "--des_prefix", help="Destination S3 folder prefix", required=True)
    parser.add_argument("-l", "--label_List", nargs='+', help="images label", required=True)
    parser.add_argument("-uid", "--user_id", help="requester user id", required=True)

    args = parser.parse_args()

    # Assign input, output files and number of lines variables from command line arguments
    des_bucket_name = args.des_bucket_name
    prefix = args.des_prefix
    label_list = args.label_List
    user_id = args.user_id


    user_info = { "destination_bucket" : des_bucket_name,
                   "destination_prefix" : prefix,
                   "user_id"    : user_id


    }


    verify_labels(label_list)
    verify_labels_quantities(label_list,user_info)