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

from keras.applications.imagenet_utils import preprocess_input




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

        url_list = []

        for i, label_name in enumerate(label_list):

            sql = "SELECT image_thumbnail_object_key FROM images WHERE label_name =  %s ;"

            cur.execute(sql,(label_name,))

            results = [r[0] for r in cur.fetchall()]

            uri_sdf = CreateTrainImageUriandLabels(results, i, label_name, label_cardinality, 0)

            for url in results:

                url_list.append(url)

            # local_train, local_test, _ = uri_sdf.randomSplit([0.005, 0.005, 0.99])

            local_train, local_test, _ = uri_sdf.randomSplit([0.8, 0.1, 0.1])

            global train_df
            global test_df

            train_df = train_df.unionAll(local_train)
            test_df = test_df.unionAll(local_test)



        # close the communication with the PostgreSQL
        cur.close()

        return url_list

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')


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

    label_list = ['Cat', 'Panda']

    label_cardinality = len(label_list)
    label_nums = list(range(label_cardinality))

    train_df = CreateTrainImageUriandLabels(['dummy'], 1, 'empty', 2, 1)
    test_df = CreateTrainImageUriandLabels(['dummy'], 0, 'empty', 2, 1)

    url_list = get_images_urls(label_list)

    print(url_list)

    from keras.applications.inception_v3 import preprocess_input
    from keras.preprocessing.image import img_to_array, load_img
    import numpy as np
    import os
    from pyspark.sql.types import StringType
    from sparkdl import KerasImageFileTransformer

    from keras.applications import InceptionV3

    # Load inception v3 (Best Image Classifier)
    model = InceptionV3(weights="imagenet")

    # Save the model
    model.save('/tmp/model-full.h5')

    # Parameters
    SIZE = (299, 299)  # Size accepted by Inception model
    IMAGES_PATH = 'datasets/image_classifier/test/'  # Images Path
    MODEL = '/tmp/model-full-tmp.h5'  # Model Path


    # Image Preprocessing
    def preprocess_keras_inceptionV3(uri):
        image = img_to_array(load_img(uri, target_size=SIZE))
        image = np.expand_dims(image, axis=0)
        return preprocess_input(image)


    # Define Spark Transformer
    transformer = KerasImageFileTransformer(inputCol="uri", outputCol="predictions",
                                            modelFile=MODEL,
                                            imageLoader=preprocess_keras_inceptionV3,
                                            outputMode="vector")



    uri_df = sqlContext.createDataFrame(url_list, StringType()).toDF("uri")

    # Get Output
    labels_df = transformer.transform(uri_df)

    # Show Output
    labels_df.show()



    #
    # train_df = train_df.filter(train_df.isDefault == 0)
    # test_df = test_df.filter(test_df.isDefault == 0)
    #
    # train_df.show()
    # test_df.show()
    #
    # # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
    # # This ensure that each of the paritions has a small size.
    # train_df = train_df.repartition(100)
    # test_df = test_df.repartition(100)
    #
    # # from keras.applications import InceptionV3
    # #
    # # model = InceptionV3(weights="imagenet")
    #
    # from keras.applications import VGG16
    #
    # model = VGG16(weights='imagenet',
    #                  include_top=False,
    #                  input_shape=(224, 224, 3))
    #
    # # model = ResNet50(weights=None,top_layer = False,input_tensor=None, input_shape=(224, 224, 3))
    #
    # model.save('/tmp/model-full.h5')  # saves to the local filesystem
    #
    # import PIL.Image
    # import numpy as np
    # from keras.applications.imagenet_utils import preprocess_input
    #
    #
    # def load_image_from_uri(local_uri):
    #
    #     response = requests.get(local_uri)
    #     # img = Image.open(BytesIO(response.content))
    #     # img = image.load_img(BytesIO(response.content), target_size=(299, 299))
    #     #
    #     img = (PIL.Image.open(BytesIO(response.content)).convert('RGB').resize((224, 224), PIL.Image.ANTIALIAS))
    #     img_arr = np.array(img).astype(np.float32)
    #     img_tnsr = preprocess_input(img_arr[np.newaxis, :])
    #
    #     print("img_tnsr: ", img_tnsr)
    #
    #     return img_tnsr
    #
    #
    #
    # stringIndexer = StringIndexer(inputCol="label_name", outputCol="categoryIndex")
    # indexed_dateset = stringIndexer.fit(train_df).transform(train_df)
    #
    #
    # # encoder = OneHotEncoder(inputCol="categoryIndex", outputCol="categoryVec")
    #
    # encoder = OneHotEncoderEstimator(inputCols=["categoryIndex"], outputCols=["categoryVec"])
    #
    # encoder_model = encoder.fit(indexed_dateset)
    #
    # image_dataset = encoder_model.transform(indexed_dateset)
    #
    # image_dataset.show()
    #
    #
    # from sparkdl.estimators.keras_image_file_estimator import KerasImageFileEstimator
    #
    #
    # estimator = KerasImageFileEstimator(inputCol="imageUri",
    #                                     outputCol="prediction",
    #                                     labelCol="categoryVec",
    #                                     imageLoader=load_image_from_uri,
    #                                     kerasOptimizer='adam',
    #                                     kerasLoss='categorical_crossentropy',
    #                                     modelFile='/tmp/model-full.h5')
    #
    # #
    # #
    # #
    # #
    # transformers = estimator.fit(image_dataset)
    # transformers.show()
    #
    # # from pyspark.ml.evaluation import BinaryClassificationEvaluator
    # # from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    # #
    # # paramGrid = (
    # #     ParamGridBuilder()
    # #         .addGrid(estimator.kerasFitParams, [{"batch_size": 16, "verbose": 0},
    # #                                             {"batch_size": 16, "verbose": 0}])
    # #         .build()
    # # )
    # # mc = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")
    # # cv = CrossValidator(estimator=estimator, estimatorParamMaps=paramGrid, evaluator=mc, numFolds=2)
    # #
    # #
    # # cvModel = cv.fit(train_df)