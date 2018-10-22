#!/usr/bin/env python2
# train_model_TF_Sample_HDFS_version.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-18-2018
# Updated Date: 09-18-2018

"""

This is a sample code to run transfer learning with SparkDL with Tensorflow backend. The source of  image data are from HDFS.

Reference:
https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6026450283250196/2720471487429801/7409402632610251/latest.html


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from pyspark.ml.image import ImageSchema
from pyspark.sql.functions import lit



banana_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Banana").withColumn("label", lit(1))


accordion_image_df = ImageSchema.readImages("hdfs://ec2-18-235-62-224.compute-1.amazonaws.com:9000/OID/Dataset/test/Accordion").withColumn("label", lit(0))


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

p_model = p.fit(train_df)

# Inspect training error
tested_df = p_model.transform(test_df.limit(10))
predictionAndLabels = tested_df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))
