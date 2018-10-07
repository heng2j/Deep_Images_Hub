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

from PIL import Image
import requests
import shutil

import numpy as np
from os.path import dirname as up


import pandas as pd


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
                print("Label " + label_name + " existed")

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
                save_to_requesting_label_to_watchlist(cur, results, user_id)

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
def save_to_requesting_label_to_watchlist(cur, label_list, user_id):

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




"""

retrieve image urls from database

Temp workflow:


"""


def get_images_urls(label_list):


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

        for label_name in label_list:

            values_list.append(label_name)


        print("values_list: ", values_list)

        sql = "SELECT full_hadoop_path , label_name  FROM images WHERE label_name IN  %(values_list)s ;"

        # execute a statement
        print('Getting image urls for requesting labels ...')
        cur.execute(sql,
                    {
                        'values_list': tuple(values_list),
                    })

        results = cur.fetchall()

        results_df = pd.DataFrame(results, columns=['image_url', 'label_name'])

        # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return results_df

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')









"""

Kick Start Traiing Process

Temp workflow:
    1. Get counts for total number of images 
    2. Evaluate the number of slaves nodes 

"""

# Invoke model training script to train model in TensorflowOnSpark with the requesting labels
def invoke_model_training(label_list,user_info):

    #TODO
    print("Invoking model training process...")
    print("Training started")




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

    label_cardinality = len(label_list)


    user_info = { "destination_bucket" : des_bucket_name,
                   "destination_prefix" : prefix,
                   "user_id"    : user_id


    }


    # # verify_labels(label_list)
    # # verify_labels_quantities(label_list,user_info)
    #
    #
    # image_urls_df = get_images_urls(label_list)
    #
    # filePath = image_urls_df.image_url[0]
    #
    #
    # # strip off the starting s3a:// from the bucket
    # bucket = os.path.dirname(str(filePath))[6:].split("/", 1)[0]
    # key = os.path.basename(str(filePath))
    # path  = filePath[6:].split("/", 1)[1:][0]
    #
    #
    # print(bucket)
    # print(key)
    # print(path)






"""

For Testing

"""




def get_images_urls_as_dataset(label_list):


    label_nums = list(range(len(label_list)))


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

        results_list = []

        for i, label_name in enumerate(label_list):

            sql = "SELECT image_thumbnail_object_key FROM images WHERE label_name =  %s ;"

            cur.execute(sql,(label_name,))

            results = [r[0] for r in cur.fetchall()]

            results_list.append(results)

        # close the communication with the PostgreSQL
        cur.close()

        # flatten the results_list
        flattened_results_list = [y for x in results_list for y in x]


        # All labels ready return True
        return flattened_results_list

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



"""
For testing purpose

"""

import PIL.Image
# import keras
# from keras.applications.imagenet_utils import preprocess_input
# from keras_preprocessing import image

def load_image_from_uri(local_uri):
  img = (get_image_array_from_S3_file(local_uri))
  img_arr = np.array(img).astype(np.float32)
  img_tnsr = preprocess_input(img_arr[np.newaxis, :])
  return img_tnsr



# this function will use boto3 on the workers directly to pull the image
# and then decode it, all in this function
def get_image_array_from_S3_file(image_url):
    import boto3
    import os

    # TODO - will need to implement exceptions handling

    s3 = boto3.resource('s3')

    # strip off the starting s3a:// from the bucket
    bucket_name = os.path.dirname(str(image_url))[6:].split("/", 1)[0]
    key = image_url[6:].split("/", 1)[1:][0]

    bucket = s3.Bucket(bucket_name)
    obj = bucket.Object(key)
    img = image.load_img(BytesIO(obj.get()['Body'].read()), target_size=(299, 299, 3))

    return img





# this function will use boto3 on the workers directly to pull the image
# and then decode it, all in this function
def download_from_S3_img_thumbnail_urls(image_url,source_type ):

    file_name = image_url.split('/')[-1]
    label_name = image_url.split('/')[-2]
    img = Image.open(requests.get(image_url, stream=True).raw)

    image_path = '/tmp/Deep_image_hub/' + source_type + '/' + label_name + '/' + file_name
    dir_name = os.path.dirname(image_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    response = requests.get(image_url, stream=True)
    with open(image_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


# Download images from URLs in dataset
def process_image_url_dataset(url_dataset,src_type):

        for url in url_dataset:
            download_from_S3_img_thumbnail_urls(url,src_type)






labels_urls_list = get_images_urls_as_dataset(label_list)

from sklearn.model_selection import train_test_split

x_train ,x_test = train_test_split(labels_urls_list,test_size=0.2)


print(x_test)


source_type = 'test'

process_image_url_dataset(x_test,source_type)



temp_train_dir = '/tmp/Deep_image_hub/train'
temp_validation_dir = '/tmp/Deep_image_hub/validation'


# Kick start training

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from __future__ import print_function
import keras
from keras.utils import to_categorical
import os
from keras.preprocessing.image import ImageDataGenerator, load_img

from keras.applications import VGG16
vgg_conv = VGG16(weights='imagenet',
                 include_top=False,
                 input_shape=(224, 224, 3))



nTrain = len(x_train)
nVal = len(x_test)



print(nTrain)
print(nVal)

datagen = ImageDataGenerator(rescale=1. / 255)
batch_size = 20

train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain, label_cardinality))

train_generator = datagen.flow_from_directory(
    temp_train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * batch_size: (i + 1) * batch_size] = features_batch
    train_labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nTrain:
        break

train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))




validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,label_cardinality))

validation_generator = datagen.flow_from_directory(
    temp_validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))





from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(label_cardinality, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=20,
                    batch_size=batch_size,
                    validation_data=(validation_features,validation_labels))





fnames = validation_generator.filenames

ground_truth = validation_generator.classes

label2index = validation_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.iteritems())


predictions = model.predict_classes(validation_features)
prob = model.predict(validation_features)


errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))

