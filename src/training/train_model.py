#!/usr/bin/env python2
# train_model.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 09-18-2018

"""

Data preprocessor is used to ....:

 Temp: ...
 TODO: ...

 1. ....


    Run with .....:

    example:



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from pyspark.ml.image import ImageSchema
from sparkdl.image import imageIO as imageIO
import pyspark.ml.linalg as spla
import pyspark.sql.types as sptyp
from pyspark.sql.functions import lit
import numpy as np

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer


from pyspark.sql.types import StructType, StructField, IntegerType,StringType,LongType,DoubleType ,FloatType
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.conf import SparkConf


from argparse import ArgumentParser
from configparser import ConfigParser
import boto3
from io import BytesIO
import psycopg2
from psycopg2 import extras
import pandas as pd
import os
from os.path import dirname as up




"""
Commonly Shared Statics

"""

sc = SparkContext(conf=SparkConf().setAppName("training inception model with user selected labels"))
executors = sc._conf.get("spark.executor.instances")
num_executors = int(executors) if executors is not None else 1

sqlContext = SQLContext(sc)

# Set up project path
projectPath = up(up(os.getcwd()))

s3_bucket_name = "s3://insight-data-images/"

database_ini_file_path = "/utilities/database/database.ini"

img_dir = "hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/flower_photos"







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

        values_list = []

        # execute a statement
        print('Getting image urls for requesting labels and append to train and test dfs ...')

        for i, label_name in enumerate(label_list):

            sql = "SELECT full_hadoop_path FROM images WHERE label_name =  %s ;"

            cur.execute(sql,(label_name,))

            results = [r[0] for r in cur.fetchall()]

            uri_sdf = CreateTrainImageUriandLabels(results, i, label_name, label_cardinality,0)

            # This is a official answer local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99])

            local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99]) # Was ([0.8, 0.1, 0.1]) before

            global train_df
            global test_df

            train_df = train_df.unionAll(local_train)
            test_df = test_df.unionAll(local_test)

        # close the communication with the PostgreSQL
        cur.close()


    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


"""

retrieve image urls from database by label_list, download the image into hdfs

Return - A pandas dataframe with two columns "image_uri" and "label" 


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

        values_list = []

        # execute a statement
        print('Getting image urls for requesting labels and append to train and test dfs ...')

        for i, label_name in enumerate(label_list):

            sql = "SELECT full_hadoop_path FROM images WHERE label_name =  %s ;"

            cur.execute(sql,(label_name,))

            results = [r[0] for r in cur.fetchall()]

            uri_sdf = CreateTrainImageUriandLabels(results, i, label_name, label_cardinality,0)

            # This is a official answer local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99])

            local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99]) # Was ([0.8, 0.1, 0.1]) before

            global train_df
            global test_df

            train_df = train_df.unionAll(local_train)
            test_df = test_df.unionAll(local_test)

        # close the communication with the PostgreSQL
        cur.close()


    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')



def CreateTrainImageUriandLabels(image_uris, label, label_name, cardinality, isDefault):
  # Create image categorical labels (integer IDs)
  local_rows = []
  for uri in image_uris:
    label_inds = np.zeros(cardinality)
    label_inds[label] = 1.0
    one_hot_vec = spla.Vectors.dense(label_inds.tolist())
    _row_struct = {"uri": uri, "one_hot_label": one_hot_vec, "label": float(label), "label_name": str(label_name), "isDefault": int(isDefault)}
    row = sptyp.Row(**_row_struct)
    local_rows.append(row)

  image_uri_df = sqlContext.createDataFrame(local_rows)
  return image_uri_df




if __name__ == '__main__':



    label_list = ['Table', 'Chair']

    label_cardinality = len(label_list)
    label_nums = list(range(label_cardinality))


    tmp_dataset_folder = "/For_Model_Training/"


    # Download images into local disk
    # Write into HDFS
    # Keep track of URIs in pandas df


    # Remove the Temp Directory in Master
    os.rmdir('new_one')


    # banana_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Banana").withColumn("label", lit(1))
    #
    # # banana_image_df = banana_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))
    #
    # accordion_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Accordion").withColumn("label", lit(0))
    #
    # # accordion_image_df = accordion_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))
    #
    # # random split train and test set data in the ratio of 99% for train, 5% for test 5% for validation
    # banana_train, banana_test, _ = banana_image_df.randomSplit([0.99, 0.005, 0.005])
    # accordion_train, accordion_test, _ = accordion_image_df.randomSplit([0.99, 0.005, 0.005])
    #
    # train_df = accordion_train.unionAll(banana_train)
    # test_df = accordion_test.unionAll(accordion_train)

    # Reading images from HDFS
    tulips_df = ImageSchema.readImages(img_dir + "/tulips").withColumn("label", lit(1))
    daisy_df = imageIO.readImagesWithCustomFn(img_dir + "/daisy", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))
    tulips_train, tulips_test, _ = tulips_df.randomSplit([0.005, 0.005, 0.99])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
    daisy_train, daisy_test, _ = daisy_df.randomSplit([0.005, 0.005, 0.99])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
    train_df = tulips_train.unionAll(daisy_train)
    test_df = tulips_test.unionAll(daisy_test)


    train_df = CreateTrainImageUriandLabels(['dummy'],1,'empty',2,1)
    test_df = CreateTrainImageUriandLabels(['dummy'],0,'empty',2,1)


    get_images_urls(label_list)

    train_df = train_df.filter(train_df.isDefault == 0)
    test_df = test_df.filter(test_df.isDefault == 0)

    train_df.show()
    test_df.show()

    # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
    # This ensure that each of the paritions has a small size.
    train_df = train_df.repartition(100)
    test_df = test_df.repartition(100)


    # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
    # This ensure that each of the paritions has a small size.
    train_df = train_df.repartition(100)
    test_df = test_df.repartition(100)

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    p_model = p.fit(train_df)

    # Inspect training error
    tested_df = p_model.transform(test_df)
    predictionAndLabels = tested_df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))








    #
    # train_df.show()
    # test_df.show()
    #
    # train_df.printSchema()
    # test_df.printSchema()

