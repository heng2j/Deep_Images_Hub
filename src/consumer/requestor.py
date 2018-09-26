#!/usr/bin/env python2
# requestor.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-21-2018
# Updated Date: 09-21-2018

"""

requestor is used by end users to make request on image classicfication models with their choices of classes.

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
        requestor.py "s3://insight-deep-images-hub/users/username_organization_id/models/packaged_food"



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