#!/usr/bin/env python3
# read_images_from_S3.py
# ---------------
# Author: Zhongheng Li
# Init Date: 10-01-2018
# Updated Date: 10-01-2018

"""

Data preprocessor is used to ....:

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
import psycopg2
import pandas as pd
from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from sparkdl.image import imageIO as imageIO
from sparkdl.image.imageIO import _decodeImage, imageSchema


from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)


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


        results_pdf = pd.DataFrame(results, columns=['image_url', 'label_name'])


        spark_df = sqlContext.createDataFrame(results_pdf)



        # close the communication with the PostgreSQL
        cur.close()

        # All labels ready return True
        return spark_df

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



# this function will use boto3 on the workers directly to pull the image
# and then decode it, all in this function
def readFileFromS3(row):
    import boto3
    import os

    s3 = boto3.client('s3')

    filePath = row.image_url
    # strip off the starting s3a:// from the bucket
    bucket = os.path.dirname(str(filePath))[6:]
    key = os.path.basename(str(filePath))

    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    contents = bytearray(body.read())
    body.close()

    if len(contents):
        try:
            decoded = _decodeImage(bytearray(contents))
            return (filePath, decoded)
        except:
            return (filePath, {"mode": "RGB", "height": 378,
                               "width": 378, "nChannels": 3,
                               "data": bytearray("ERROR")})


label_list = ['Table', 'Chair','Drawer']

spark_df = get_images_urls(label_list)



for row in spark_df.take(1):
    print (row)

# rows_df is a dataframe with a single string column called "image_url" that has the full s3a filePath
# Running rows_df.rdd.take(2) gives the output
# [Row(image_url=u's3a://mybucket/14f89051-26b3-4bd9-88ad-805002e9a7c5'),
# Row(image_url=u's3a://mybucket/a47a9b32-a16e-4d04-bba0-cdc842c06052')]

# farm out our images to the workers with a map and get back a dataframe
# schema = StructType([StructField("filePath", StringType(), False), StructField("image", imageSchema)])
#
# image_df = (
#     rows_df
#         .rdd
#         .map(readFileFromS3)
#         .toDF(schema)
# )
#
#
# image_df.show()

# (
#     image_df
#         .write
#         .format("parquet")
#         .mode("overwrite")
#         .option("compression", "gzip")
#         .save("s3://my_bucket/images.parquet")
# )

