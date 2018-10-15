#!/usr/bin/env python2
# train_model.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-29-2018
# Updated Date: 10-1-2018

"""

:

 Temp: ...
 TODO: ...

 1. ....


    Run with .....:

    example:



"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from argparse import ArgumentParser
from configparser import ConfigParser
import boto3
from io import BytesIO
import psycopg2
from psycopg2 import extras
import pandas as pd
import os
from os.path import dirname as up

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator

from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit
from sparkdl.image import imageIO
import pyspark.ml.linalg as spla
import pyspark.sql.types as sptyp
import numpy as np

from pyspark.sql.types import StructType, StructField, IntegerType,StringType,LongType,DoubleType ,FloatType
from pyspark.sql import SQLContext
from pyspark.context import SparkContext
from pyspark.conf import SparkConf



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





# def CreateTrainImageUriandLabels(image_uris, label, label_name, cardinality, isDefault):
#   # Create image categorical labels (integer IDs)
#   local_rows = []
#   for uri in image_uris:
#     label_inds = np.zeros(cardinality)
#     label_inds[label] = 1.0
#     one_hot_vec = spla.Vectors.dense(label_inds.tolist())
#     _row_struct = {"uri": uri, "one_hot_label": one_hot_vec, "label": float(label), "label_name": str(label_name), "isDefault": int(isDefault)}
#     row = sptyp.Row(**_row_struct)
#     local_rows.append(row)
#
#   image_uri_df = sqlContext.createDataFrame(local_rows)
#   return image_uri_df



# banana_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Banana").withColumn("label", lit(1))
#
# # banana_image_df = banana_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))
#
# accordion_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Accordion").withColumn("label", lit(0))
#
# # accordion_image_df = accordion_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))





#
# banana_train, banana_test, _ = banana_image_df.randomSplit([0.99, 0.005, 0.005])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
# accordion_train, accordion_test, _ = accordion_image_df.randomSplit([0.99, 0.005, 0.005])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
#
# train_df = accordion_train.unionAll(banana_train)
# test_df = accordion_test.unionAll(accordion_train)
#
# # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# # This ensure that each of the paritions has a small size.
# train_df = train_df.repartition(100)
# test_df = test_df.repartition(100)




# # move to a permanent place for future use
# dbfs_model_full_path = 'dbfs:/models/model-full.h5'
# dbutils.fs.cp('file:/tmp/model-full.h5', dbfs_model_full_path)

from PIL import Image

import requests
from io import BytesIO
from keras.applications.imagenet_utils import preprocess_input
from keras_preprocessing import image
#
# def load_image_from_uri(local_uri):
#
#
#     # print("local_uri: " , local_uri)
#
#     response = requests.get(local_uri)
#     #img = Image.open(BytesIO(response.content))
#     img = image.load_img(BytesIO(response.content), target_size=(299, 299))
#
#     # print("img type: ", type(img))
#
#     img_arr = np.array(img).astype(np.float32)
#
#     # print("img_arr shape: ", img_arr.shape)
#     img_tnsr = preprocess_input(img_arr[np.newaxis, :])
#
#
#     return img_tnsr



#
# def get_image_array_from_S3_file(image_url):
#     import boto3
#     import os
#
#     # TODO - will need to implement exceptions handling
#
#     s3 = boto3.resource('s3')
#
#     # strip off the starting s3a:// from the bucket
#     bucket_name = os.path.dirname(str(image_url))[6:].split("/", 1)[0]
#     key = image_url[6:].split("/", 1)[1:][0]
#
#     bucket = s3.Bucket(bucket_name)
#     obj = bucket.Object(key)
#     img = image.load_img(BytesIO(obj.get()['Body'].read()), target_size=(299, 299))
#
#     return img





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
            _row_struct = {"imageUri": uri, "one_hot_label": one_hot_vec, "label": float(label),
                           "label_name": str(label_name), "isDefault": int(isDefault)}
            row = sptyp.Row(**_row_struct)
            local_rows.append(row)

        image_uri_df = sqlContext.createDataFrame(local_rows)
        return image_uri_df


    label_cardinality = 2

    label_list = ['Table', 'Chair']

    label_cardinality = len(label_list)
    label_nums = list(range(label_cardinality))

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

    # from keras.applications import InceptionV3
    #
    # model = InceptionV3(weights="imagenet")

    from keras.applications import VGG16

    model = VGG16(weights='imagenet',
                     include_top=False,
                     input_shape=(224, 224, 3))

    # model = ResNet50(weights=None,top_layer = False,input_tensor=None, input_shape=(224, 224, 3))

    model.save('/tmp/model-full.h5')  # saves to the local filesystem

    import PIL.Image
    import numpy as np
    from keras.applications.imagenet_utils import preprocess_input


    def load_image_from_uri(local_uri):

        response = requests.get(local_uri)
        # img = Image.open(BytesIO(response.content))
        # img = image.load_img(BytesIO(response.content), target_size=(299, 299))
        #
        img = (PIL.Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224), PIL.Image.ANTIALIAS))
        img_arr = np.array(img).astype(np.float32)
        img_tnsr = preprocess_input(img_arr[np.newaxis, :])

        print("img_tnsr: ", img_tnsr)

        return img_tnsr



    stringIndexer = StringIndexer(inputCol="label_name", outputCol="categoryIndex")
    indexed_dateset = stringIndexer.fit(train_df).transform(train_df)


    # encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")

    encoder = OneHotEncoderEstimator(inputCols=["categoryIndex"], outputCols=["categoryVec"])

    encoder_model = encoder.fit(indexed_dateset)

    image_dataset = encoder_model.transform(indexed_dateset)

    image_dataset.show()


    from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator


    estimator = KerasImageFileEstimator(inputCol="imageUri",
                                        outputCol="prediction",
                                        labelCol="categoryVec",
                                        imageLoader=load_image_from_uri,
                                        kerasOptimizer='adam',
                                        kerasLoss='categorical_crossentropy',
                                        modelFile='/tmp/model-full.h5')

    #
    #
    #
    #
    transformers = estimator.fit(image_dataset)
    transformers.show()

    # from pyspark.ml.evaluation import BinaryClassificationEvaluator
    # from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    #
    # paramGrid = (
    #     ParamGridBuilder()
    #         .addGrid(estimator.kerasFitParams, [{"batch_size": 16, "verbose": 0},
    #                                             {"batch_size": 16, "verbose": 0}])
    #         .build()
    # )
    # mc = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    # cv = CrossValidator(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=mc, numFolds=2)
    #
    #
    # cvModel = cv.fit(train_df)