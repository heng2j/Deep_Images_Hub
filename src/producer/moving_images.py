# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# moving_images.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-27-2018
# Updated Date: 09-31-2018

"""
moving_images:

 Temp: Take images from S3 bucket
 TODO: Accept images from user submissions from iOS devices

 1. Intake image
 2. Classify Label  - Temp with dictionary / TODO with WordNet
 3. Take Geoinfo - Temp with auto generated lat & lon / TODO with geographical info from image metadata
 4. Put the image into an existing folder with existing label. Temp - Create new folder if label is not existed.
 4. Insert image metadata into PostgreSQL database: image path on S3, label, category, subcategory, geometry, city, country, timestamp



    Current default S3 Bucket: s3://insight-data-images/Entity

    Run with .....:

    example:
            python producer.py --src_bucket_name "insight-data-images" --src_prefix "Entity/food/packaged_food/protein_bar/samples/" --des_bucket_name "insight-deep-images-hub"  --label_name "Think_thin_high_protein_caramel_fudge" --lon -73.935242 --lat 40.730610 --batch_id 1 --user_id 1


"""


from pyspark.ml.image import ImageSchema


image_df = ImageSchema.readImages("/OID/Dataset/test/Accordion")


image_df.show()

from sparkdl.image.imageIO import _decodeImage, imageSchema


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


# rows_df is a dataframe with a single string column called "image_url" that has the full s3a filePath
# Running rows_df.rdd.take(2) gives the output
# [Row(image_url=u's3a://mybucket/14f89051-26b3-4bd9-88ad-805002e9a7c5'),
# Row(image_url=u's3a://mybucket/a47a9b32-a16e-4d04-bba0-cdc842c06052')]

# farm out our images to the workers with a map and get back a dataframe
schema = StructType([StructField("filePath", StringType(), False), StructField("image", imageSchema)])

image_df = (
    rows_df
        .rdd
        .map(readFileFromS3)
        .toDF(schema)
)

(
    image_df
        .write
        .format("parquet")
        .mode("overwrite")
        .option("compression", "gzip")
        .save("s3://my_bucket/images.parquet")
)