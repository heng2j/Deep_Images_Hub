#!/usr/bin/env python2
# postTraining.py
# ---------------
# Author: Zhongheng Li
# Init Date: 10-05-2018
# Updated Date: 10-10-2018

"""

requester is used by end users to make request on image classicfication models with their choices of classes.

 Temp: ...
 TODO: ...

 Given: user_id, classes_list, destination_bucket



    Run with .....:

    example:
        requester.py "s3://insight-deep-images-hub/users/username_organization_id/models/packaged_food"

        python requester.py --des_bucket_name "insight-deep-images-hub" --des_prefix "user/userid/model/" --label_List 'Apple' 'Banana' 'protein_bar'  --user_id 2


"""

from __future__ import print_function
from argparse import ArgumentParser
from configparser import ConfigParser
import os
from io import BytesIO
import psycopg2
from psycopg2 import extras
import boto3
from PIL import Image
import requests
import shutil
import datetime
import numpy as np
from os.path import dirname as up

import pandas as pd


"""
  Commonly Shared Statics

  """

# Set up project path
# projectPath = up(up(os.getcwd()))
projectPath = os.getcwd()

s3_bucket_name = "insight-deep-images-hub"

database_ini_file_path = "/Deep_Images_Hub/utilities/database/database.ini"

"""

config Database

"""


def config(filename=projectPath + database_ini_file_path, section='postgresql'):
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
    Send training results to S3 bucket to both S3 model pool bucket and user's bucket

"""

# This upload files in directory to S3 code is referenced from https://www.developerfiles.com/upload-files-to-s3-with-python-keeping-the-original-folder-structure/
def upload_files(path,des_prefix):
    s3 = boto3.resource('s3', region_name='us-east-1')
    bucket = s3.Bucket(s3_bucket_name)

    for subdir, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, 'rb') as data:
                if not '.DS_Store' in full_path:
                    print("Putting file ", des_prefix + full_path[len(path) + 1:])
                    bucket.put_object(Key=des_prefix+full_path[len(path) + 1:], Body=data, ACL='public-read')

def copy_training_results_to_S3(user_id,model_id,source_path,des_prefix,user_des_prefix):

    print("Copying training results to S3...")
    # upload results to model pool
    upload_files(source_path,des_prefix)

    # upload results to user's S3 bucket
    upload_files(source_path,user_des_prefix)


def save_results_to_db(request_number, save_path):
    sql = """

    UPDATE training_records
	SET 							  
    saved_model_path = %s,
	creation_date	= %s,			  
    WHERE model_id = %s;								  


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

        # writing image info into the database
        # execute a statement
        print('writing image batch info into the database...')
        cur.execute(sql, (save_path, datetime.datetime.now(), request_number))

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
    parser.add_argument("-tn", "--training_request_number", required=True,
                    help="this model training request number")
    parser.add_argument("-uid", "--user_id", help="requester user id", required=True)

    args = parser.parse_args()

    # Assign input, output files and number of lines variables from command line arguments
    user_id = args.user_id
    model_id = args.training_request_number

    des_prefix = "trained_model/" + datetime.datetime.today().strftime('%Y-%m-%d') + '/' + model_id + '/'

    user_des_prefix = "user/" + user_id + '/training_results/' + datetime.datetime.today().strftime('%Y-%m-%d') + '/'


    source_path = '/tmp/Deep_image_hub_Model_Training'


    copy_training_results_to_S3(user_id, model_id, source_path, des_prefix, user_des_prefix)


    # Save results to database
    print("Saving results to database...")
    save_results_to_db(model_id, des_prefix)

    print("Post training process completed")


