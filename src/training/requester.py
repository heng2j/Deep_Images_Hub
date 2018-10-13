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
from argparse import ArgumentParser
from configparser import ConfigParser
import os
import psycopg2
from PIL import Image
import requests
import shutil
import logging
import pandas as pd
import sys


log = logging.getLogger(__name__)

"""
Commonly Shared Statics

"""

# Set up project path
# projectPath = up(up(os.getcwd()))
projectPath = os.getcwd()

s3_bucket_name = "s3://insight-data-images/"

database_ini_file_path = "/Deep_Images_Hub/utilities/database/database.ini"





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
        # print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        #print('Verifying if the labels existed in the database...')
        log.info('Verifying if the labels existed in the database...')
        for label_name in label_list:

            #TODO -  augmented SQL statement
            sql = "SELECT count(label_name)  FROM labels WHERE label_name = '" + label_name +"' ;"
            # print("sql: ", sql)
            log.info("sql: ", sql)

            # verify if label exist in the database
            # execute a statement
            cur.execute(sql)

            result_count = cur.fetchone()[0]

            if result_count == 1:
                # print("Label " + label_name + " existed")
                log.info("Label " + label_name + " existed")

            else:
                #print("Label '" + label_name +"' doesn't exist")
                log.info("Label '" + label_name +"' doesn't exist")
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
            #print('Database connection closed.')
            log.info('Database connection closed.')





# Verify if the labels currently have enough images
def verify_labels_quantities(label_list,user_id):

    # user_id = user_info['user_id']

    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        #print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        values_list = []

        for label_name in label_list:

            values_list.append(label_name)


        sql = "SELECT label_name FROM labels WHERE label_name IN %(values_list)s AND image_count < 100;"



        # execute a statement
        #print('Getting image urls for requesting labels ...')
        log.info('Getting image urls for requesting labels ...')

        cur.execute(sql,
                    {
                        'values_list': tuple(values_list),
                    })

        results = cur.fetchall()

        # The returning results are the labels that doesn't have enough images
        if results:
            #print("The following labels does not have enough images:")
            log.info("The following labels does not have enough images:")
            #print(results)
            log.info(results)

            #print("These labels will save into the requesting_label_watchlist table")
            log.info("These labels will save into the requesting_label_watchlist table")

            # TODO - Save labels into the requesting_label_watchlist table
            save_to_requesting_label_to_watchlist(cur, results, user_id)

            # commit changes
            conn.commit()

            # close the communication with the PostgreSQL
            cur.close()
            #print('Database connection closed.')
            log.info('Database connection closed.')

            return False


        else:
            #print("All labels are ready for training :) ")
            log.info("All labels are ready for training :) ")

            # close the communication with the PostgreSQL
            cur.close()
            #print('Database connection closed.')
            log.info('Database connection closed.')

            return True


    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            #print('Database connection closed.')
            log.info('Database connection closed.')





# Save labels into the requesting_label_watchlist table
def save_to_requesting_label_to_watchlist(cur, label_list, user_id):

    #print('Saving labels into the requesting_label_watchlist table...')
    log.info('Saving labels into the requesting_label_watchlist table...')
    for label_name in label_list:

        # TODO -  augmented SQL statement

        sql = " INSERT INTO requesting_label_watchlist (label_name, user_ids,last_requested_userid, new_requested_date ) VALUES \
        ( '"+ label_name[0] +"',ARRAY[" + user_id + "], " + user_id + ", (SELECT NOW()) ) ON CONFLICT (label_name)\
        DO UPDATE \
        SET user_ids = array_append(requesting_label_watchlist.user_ids, "+ user_id +"),\
        last_requested_userid = " + user_id + " \
        WHERE requesting_label_watchlist.label_name = '" + label_name[0] + "';"

        #print("sql: ", sql)
        log.info("sql: ", sql)

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
        # print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        values_list = []

        for label_name in label_list:

            values_list.append(label_name)


        print("values_list: ", values_list)
        log.info("values_list: ", values_list)

        sql = "SELECT full_hadoop_path , label_name  FROM images WHERE label_name IN  %(values_list)s ;"

        # execute a statement
        #print('Getting image urls for requesting labels ...')
        log.info('Getting image urls for requesting labels ...')
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
            #print('Database connection closed.')
            log.info('Database connection closed.')








def get_images_urls_as_dataset(label_list):


    label_nums = list(range(len(label_list)))


    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        #print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')
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
            # print('Database connection closed.')
            log.info('Database connection closed.')




# this function will use boto3 on the workers directly to pull the image
# and then decode it, all in this function
def download_from_S3_img_thumbnail_urls(image_url):

    file_name = image_url.split('/')[-1]
    label_name = image_url.split('/')[-2]
    img = Image.open(requests.get(image_url, stream=True).raw)

    image_path = '/tmp/Deep_image_hub_Model_Training/dataset/' + label_name + '/' + file_name
    dir_name = os.path.dirname(image_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    response = requests.get(image_url, stream=True)
    with open(image_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


# Download images from URLs in dataset
def download_image_dataset_from_image_urls(url_dataset):

        for url in url_dataset:
            download_from_S3_img_thumbnail_urls(url)



def get_image_counters_for_labels(label_list):

    label_nums = list(range(len(label_list)))


    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        # print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')
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
            #print('Database connection closed.')
            log.info('Database connection closed.')


"""

Kick Start Traiing Process

Temp workflow:
    1. Get counts for total number of images 
    2. Evaluate the number of slaves nodes 

"""

# Invoke model training script to train model in TensorflowOnSpark with the requesting labels
def invoke_model_training(label_list,user_id):

    # print("Invoking model training process...")
    log.info("Invoking model training process...")

    import shutil

    #print("Removing old training data and model if exist...")
    log.info("Removing old training data and model if exist...")

    # Remove previous trained data if exist
    if os.path.exists('/tmp/Deep_image_hub_Model_Training/model/'):
        shutil.rmtree("/tmp/Deep_image_hub_Model_Training/model")

    if os.path.exists('/tmp/Deep_image_hub_Model_Training/dataset/'):
        shutil.rmtree("/tmp/Deep_image_hub_Model_Training/dataset")



    # print ("Generating the list of urls from requesting lables...")
    log.info("Generating the list of urls from requesting lables...")

    labels_urls_list = get_images_urls_as_dataset(label_list)

    # print ("Downloading images from url list...")
    log.info("Downloading images from url list...")

    download_image_dataset_from_image_urls(labels_urls_list)

    # print ("Download Completed")
    log.info("Download Completed")


    # print("Inserting traiing records in database...")
    log.info("Inserting traiing records in database...")


    sql = """

    INSERT INTO training_records (label_names, image_counts_for_labels, initial_requested_user_id, creation_date ) 
    VALUES 
     
    (ARRAY%s,
     
    (SELECT ARRAY(  
    with x (id_list) as (
      values (ARRAY%s)
    )
    select  image_count
    from labels, x
    where label_name = any (x.id_list)
    order by array_position(x.id_list, label_name)
    )
    )
     ,%s, (SELECT NOW()))
     
    RETURNING model_id;

    """ % (label_list,label_list,user_id)


    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config()

        # connect to the PostgreSQL server
        # print('Connecting to the PostgreSQL database...')
        log.info('Connecting to the PostgreSQL database...')

        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        cur.execute(sql)

        # commit the changes to the database
        conn.commit()

        model_id = cur.fetchone()[0]

        # close the communication with the PostgreSQL
        cur.close()

        return model_id

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            #print('Database connection closed.')
            log.info('Database connection closed.')



if __name__ == '__main__':

    # Set up argument parser
    parser = ArgumentParser()
    parser.add_argument("-l", "--label_List", nargs='+', help="images label", required=True)
    parser.add_argument("-uid", "--user_id", help="requester user id", required=True)

    args = parser.parse_args()

    label_list = args.label_List
    user_id = args.user_id

    label_cardinality = len(label_list)

    # Verify if the labels are existed in the Database
    if not verify_labels(label_list):
        exit()

    # Verify if all the requesting labels are having enough images to train.
    if not verify_labels_quantities(label_list,user_id):
        exit()


    model_id = invoke_model_training(label_list, user_id)

    sys.exit(model_id)







