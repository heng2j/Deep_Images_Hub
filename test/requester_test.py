#!/usr/bin/env python3
# requester_test.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-21-2018
# Updated Date: 10-16-2018


from __future__ import print_function
from argparse import ArgumentParser
from configparser import ConfigParser
import os
from io import BytesIO
import matplotlib.pyplot as plt
import psycopg2
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


    image_urls_df = get_images_urls(label_list)

    filePath = image_urls_df.image_url[0]


    # strip off the starting s3a:// from the bucket
    bucket = os.path.dirname(str(filePath))[6:].split("/", 1)[0]
    key = os.path.basename(str(filePath))
    path  = filePath[6:].split("/", 1)[1:][0]


    print(bucket)
    print(key)
    print(path)



    """
    
    Functions For Testing
    
    """



    def get_images_urls(label_list):


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

                sql = "SELECT full_hadoop_path FROM images WHERE label_name =  %s ;"

                cur.execute(sql,(label_name,))

                results = [r[0] for r in cur.fetchall()]

                print(results)

                results_list.append(results)

            # close the communication with the PostgreSQL
            cur.close()

            # All labels ready return True
            return results_list

        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if conn is not None:
                conn.close()
                print('Database connection closed.')



    """
    For testing retrieve images from S3 urls 
    
    """

    import PIL.Image
    import keras
    from keras.applications.imagenet_utils import preprocess_input
    from keras_preprocessing import image

    def load_image_from_uri(local_uri):


      # img = (PIL.Image.open(local_uri).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))

      img = (get_image_array_from_S3_file(local_uri))

      img_arr = np.array(img).astype(np.float32)
      img_tnsr = preprocess_input(img_arr[np.newaxis, :])
      return img_tnsr



    def load_image_from_uri_local(local_uri):
      img = (PIL.Image.open(local_uri).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))


      plt.figure(0)
      plt.imshow(img)
      plt.title('Sample Image from S3')
      plt.pause(0.05)



      img_arr = np.array(img).astype(np.float32)
      img_tnsr = preprocess_input(img_arr[np.newaxis, :])
      return img_tnsr



    # this function will use boto3 on the workers directly to pull the image
    # and then decode it, all in this function
    def get_image_array_from_S3_file(image_url):
        import boto3
        import os


        s3 = boto3.resource('s3')

        # strip off the starting s3a:// from the bucket
        bucket_name = os.path.dirname(str(image_url))[6:].split("/", 1)[0]
        key = image_url[6:].split("/", 1)[1:][0]

        bucket = s3.Bucket(bucket_name)
        obj = bucket.Object(key)
        img = image.load_img(BytesIO(obj.get()['Body'].read()), target_size=(299, 299))

        return img



    labels_urls_list = get_images_urls(label_list)


    image_url = labels_urls_list[0][0]

    print(image_url)

    img = get_image_array_from_S3_file(image_url)

    print("img from s3 type: ", type(img))

    plt.figure(0)
    plt.imshow(img)
    plt.title('Sample Image from S3')
    plt.pause(0.05)


    img_tnsr = load_image_from_uri(image_url)

    print (img_tnsr)

    local_image_path = projectPath + "/data/images/dummy_dataset_1000/car/2008_000085.jpg"


    load_image_from_uri_local(local_image_path)

    for label in labels_urls_list:
        for url in label:
            img_tnsr = load_image_from_uri(url)
            print(type(img_tnsr))
