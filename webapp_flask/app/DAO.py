#!/usr/bin/env python3
# DAO.py
# ---------------
# Author: Zhongheng Li
# Init Date: 10-02-2018
# Updated Date: 10-03-2018

"""

DAO is used to ....:

 Temp: ...
 TODO: ...

 1. Get labels from requester
 2. Retrieve image_df from db and


    Run with .....:

    example:



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from argparse import ArgumentParser
from configparser import ConfigParser
import os
from os.path import dirname as up
import psycopg2
import pandas as pd
import numpy as np




"""
Commonly Shared Statics

"""

# Set up project path
projectPath = up(os.getcwd())

s3_bucket_name = "s3://insight-deep-images-hub/"

# database_ini_file_path = "/app/util/database.ini"

database_ini_file_path = "/Project/utilities/database/database.ini"

s3_url_prefix = "http://s3.amazonaws.com/"


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

Quering functions

"""

# Select 9 most recent image batches
def get_featured_batches():


    sql = """

        SELECT (SELECT label_name FROM images WHERE batch_id in (ib.batch_id) LIMIT 1 ) AS label_name , ib.submitted_count, ib.on_board_date, pl.city, pl.neighbourhood, pl.geometry, (SELECT image_thumbnail_object_key FROM images WHERE batch_id in (ib.batch_id) LIMIT 1 ) AS sample_image
        FROM images_batches AS ib
        INNER JOIN places as pl
        ON ib.place_id = pl.place_id
        WHERE ib.submitted_count > 0
        ORDER BY ib.on_board_date  DESC
        LIMIT 9;

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

        # execute a statement
        print('Getting image urls for requesting labels ...')
        cur.execute(sql)

        results = cur.fetchall()


        results_pdf = pd.DataFrame(results, columns=['label_name','submitted_count','on_board_date','city','neighbourhood','geometry','sample_image'])

        featured_batches = []

        for index, row in results_pdf.iterrows():

            batch_details = {}
            batch_details['label_name'] = row.label_name
            batch_details['submitted_count'] = row.submitted_count
            batch_details['on_board_date'] = row.on_board_date
            batch_details['city'] = row.city
            batch_details['neighbourhood'] = row.neighbourhood
            batch_details['geometry'] = row.geometry


            # sample_image_url = row.sample_image.replace("s3a://", s3_url_prefix)
            batch_details['sample_image'] = row.sample_image

            featured_batches.append(batch_details)


        # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return featured_batches

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


# Get all labels with images count > 0
def get_all_labels():


    sql_get_all_labels = """

        SELECT l.label_name, l.parent_name,l.image_count, l.updated_date, (SELECT image_thumbnail_object_key FROM images WHERE label_name in (l.label_name) ORDER BY random() LIMIT 1 ) AS sample_image
        FROM labels AS l
        WHERE l.image_count > 0
        ORDER BY l.label_name ; 


	
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

        # execute a statement
        print('Getting all labels with images count > 0 ...')
        cur.execute(sql_get_all_labels)

        results = cur.fetchall()


        results_pdf = pd.DataFrame(results,
                                   columns=['label_name', 'parent_name', 'image_count', 'updated_date','sample_image'])

        parent_labels = []

        for index, row in results_pdf.iterrows():

            label_details = {}
            label_details['label_name'] = row.label_name
            label_details['parent_name'] = row.parent_name
            # label_details['children'] = row.children
            label_details['image_count'] = row.image_count
            label_details['updated_date'] = row.updated_date
            label_details['sample_image'] = row.sample_image

            parent_labels.append(label_details)

        # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return parent_labels

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



# Get all parent labels with children has images
def get_parent_labels():

    sql_parent_labels = """

        SELECT DISTINCT l.parent_name 
        FROM labels AS l
        WHERE l.image_count > 0;

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

        # execute a statement
        print('Getting all parent labels with children has images ...')
        cur.execute(sql_parent_labels)

        results_list = sorted([result[0] for result in cur.fetchall()])

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




# featured_batches = get_featured_batches()
#
# print(featured_batches)
# print(type(featured_batches))


