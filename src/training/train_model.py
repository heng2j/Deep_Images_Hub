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
from pyspark.sql.functions import lit
from sparkdl.image import imageIO as imageIO


#
# tulips_df = imageIO.readImagesWithCustomFn("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/" + "flower_photos/tulips", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))
# daisy_df = imageIO.readImagesWithCustomFn("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/" + "flower_photos/daisy", decode_f=imageIO.PIL_decode).withColumn("label", lit(0))
# tulips_train, tulips_test, _ = tulips_df.randomSplit([0.005, 0.005, 0.99])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
# daisy_train, daisy_test, _ = daisy_df.randomSplit([0.005, 0.005, 0.99])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
# train_df = tulips_train.unionAll(daisy_train)
# test_df = tulips_test.unionAll(daisy_test)
#
# # Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# # This ensure that each of the paritions has a small size.
# train_df = train_df.repartition(100)
# test_df = test_df.repartition(100)
#
#
# from pyspark.ml.classification import LogisticRegression
# from pyspark.ml import Pipeline
# from sparkdl import DeepImageFeaturizer
#
# featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
# lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
# p = Pipeline(stages=[featurizer, lr])
#
# p_model = p.fit(train_df)
#
#
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#
# tested_df = p_model.transform(test_df)
# evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# print("Test set accuracy = " + str(evaluator.evaluate(tested_df.select("prediction", "label"))))



banana_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Banana").withColumn("label", lit(1))


# banana_image_df = banana_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))

accordion_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Accordion").withColumn("label", lit(0))

# accordion_image_df = accordion_image_df.withColumn("prefix", lit('Entity/data/food/fruit/'))


banana_train, banana_test, _ = banana_image_df.randomSplit([0.99, 0.005, 0.005])  # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)
accordion_train, accordion_test, _ = accordion_image_df.randomSplit([0.99, 0.005, 0.005])     # use larger training sets (e.g. [0.6, 0.4] for non-community edition clusters)

train_df = accordion_train.unionAll(banana_train)
test_df = accordion_test.unionAll(accordion_train)

# Under the hood, each of the partitions is fully loaded in memory, which may be expensive.
# This ensure that each of the paritions has a small size.
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)






from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

p_model = p.fit(train_df)    # train_images_df is a dataset of images and labels

# Inspect training error
tested_df = p_model.transform(test_df.limit(10))
predictionAndLabels = tested_df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
