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


from pyspark.sql import types

from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.conf import SparkConf


from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from sparkdl.image import imageIO as imageIO
from pyspark.sql import DataFrame, SparkSession


import requests
from argparse import ArgumentParser
from configparser import ConfigParser
import boto3
from io import BytesIO
import psycopg2
from psycopg2 import extras
import pandas as pd
import os
from os.path import dirname as up

from keras.applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
from sparkdl.image.imageIO import imageArrayToStruct

from pyspark.sql.types import StructType, StructField, IntegerType,StringType,LongType,DoubleType ,FloatType , BinaryType , Row
import pyspark.sql.types as sptyp

import PIL.Image


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

            sql = "SELECT image_thumbnail_object_key FROM images WHERE label_name =  %s ;"

            cur.execute(sql,(label_name,))

            results = [r[0] for r in cur.fetchall()]

            uri_sdf = CreateTrainImageUriandLabels(results, i, label_name, label_cardinality, 0)

            # local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99])
            # local_train, local_test, _ = uri_sdf.randomSplit([0.8, 0.1, 0.1])

            local_train, local_test, _ = uri_sdf.randomSplit([0.8, 0.1, 0.1])

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



# def load_image_from_uri(local_uri):
#
#
#     response = requests.get(local_uri)
#     #img = Image.open(BytesIO(response.content))
#     img = image.load_img(BytesIO(response.content), target_size=(299, 299))
#
#     # print("img type: ", type(img))
#
#     img_arr = np.array(img).astype(np.float32)
#     #
#     # print("img_arr shape: ", img_arr.shape)
#     # img_tnsr = preprocess_input(img_arr[np.newaxis, :])
#     #
#     # print("img_tnsr.shape", img_tnsr.shape)
#
#
#     return img_arr


def load_image_from_uri(local_url):

    response = requests.get(local_url)
    img = (PIL.Image.open(BytesIO(response.content)).convert('RGB').resize((299, 299), PIL.Image.ANTIALIAS))
    img_arr = np.array(img).astype(np.float32)
    img_tnsr = preprocess_input(img_arr[np.newaxis, :])

    return img_tnsr


def create_image_dataframe(row):

    img_array = load_image_from_uri(row.uri)

    image_dataframe = imageArrayToStruct(img_array)

    # updated Mode to be 16 _OcvType(name="CV_8UC3", ord=16, nChannels=3, dtype="uint8"), : reference https://github.com/databricks/spark-deep-learning/blob/master/python/sparkdl/image/imageIO.py
    d = image_dataframe.asDict()
    d['mode'] = 16
    new_row = Row(**d)

    return new_row , row.label



if __name__ == '__main__':

    from pyspark.ml.image import ImageSchema
    from pyspark.sql.functions import lit
    from sparkdl.image import imageIO
    from pyspark.sql.functions import col, asc
    import pyspark.ml.linalg as spla
    import pyspark.sql.types as sptyp
    import numpy as np


    def CreateTrainImageUriandLabels(image_uris, label, label_name, cardinality, isDefault):
        # Create image categorical labels (integer IDs)
        local_rows = []
        for uri in image_uris:
            label_inds = np.zeros(cardinality)
            label_inds[label] = 1.0
            one_hot_vec = spla.Vectors.dense(label_inds.tolist())
            _row_struct = {"uri": uri, "one_hot_label": one_hot_vec, "label": int(label),
                           "label_name": str(label_name), "isDefault": int(isDefault)}
            row = sptyp.Row(**_row_struct)
            local_rows.append(row)

        image_uri_df = sqlContext.createDataFrame(local_rows)
        return image_uri_df


    label_cardinality = 2

    label_list = ['Tap', 'Teapot']

    label_cardinality = len(label_list)
    label_nums = list(range(label_cardinality))




    banana_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Banana").withColumn("label", lit(1))


    # banana_image_df = banana_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))

    accordion_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Accordion").withColumn("label", lit(0))

    # accordion_image_df = accordion_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))


    banana_train, banana_test, _ = banana_image_df.randomSplit([0.99, 0.005, 0.005])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
    accordion_train, accordion_test, _ = accordion_image_df.randomSplit([0.99, 0.005, 0.005])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)

    train_df = accordion_train.unionAll(banana_train)
    test_df = accordion_test.unionAll(accordion_train)

    train_df.show()
    train_df.printSchema()

    train_df.select("image.*").show()

    #
    #
    # from pyspark.ml.classification import LogisticRegression
    # from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    # from pyspark.ml import Pipeline
    # from sparkdl import DeepImageFeaturizer
    #
    # featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    # lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    # p = Pipeline(stages=[featurizer, lr])
    #
    # p_model = p.fit(train_df)
    #
    # # Inspect training error
    # tested_df = p_model.transform(test_df.limit(10))
    # predictionAndLabels = tested_df.select("prediction", "label")
    # evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    # print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))



    # train_df.select(train_df.columns[:1]).take(5).show()


    #
    # # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
    # # This ensure that each of the paritions has a small size.
    train_df = train_df.repartition(100)
    test_df = test_df.repartition(100)
    #
    #



    train_df = CreateTrainImageUriandLabels(['dummy'], 1, 'empty', 2, 1)
    test_df = CreateTrainImageUriandLabels(['dummy'], 0, 'empty', 2, 1)


    get_images_urls(label_list)

    train_df = train_df.filter(train_df.isDefault == 0)
    test_df = test_df.filter(test_df.isDefault == 0)

    train_df.show()
    test_df.show()







    # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
    # This ensure that each of the paritions has a small size.
    train_df = train_df.repartition(100)
    test_df = test_df.repartition(100)

    imageSchema = StructType([
                                StructField("origin", StringType(), True),
                              StructField("height", IntegerType(), False),
                              StructField("width", IntegerType(), False),
                              StructField("nChannels", IntegerType(), False),
                              StructField("mode", IntegerType(), False),
                              StructField("data", BinaryType(), False)])



    schema = StructType([StructField("image", imageSchema), StructField("label", IntegerType(), False)])


    image_df = (
        train_df
            .rdd
            .map(create_image_dataframe)
            .toDF(schema)
    )

    image_df.show()
    image_df.printSchema()
    image_df.select("image.*").show()








    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import MulticlassClassificationEvaluator
    from pyspark.ml import Pipeline
    from sparkdl import DeepImageFeaturizer

    featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
    lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
    p = Pipeline(stages=[featurizer, lr])

    p_model = p.fit(image_df)

    # Inspect training error
    tested_df = p_model.transform(test_df.limit(10))
    predictionAndLabels = tested_df.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
