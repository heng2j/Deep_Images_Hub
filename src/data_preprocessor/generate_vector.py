# -*- coding: utf-8 -*-
#!/usr/bin/env python2
# generate_vector.py
# ---------------
# Author: Zhongheng Li
# Init Date: 09-27-2018
# Updated Date: 09-27-2018

"""

generate_vector takes images and :

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




def load_headless_pretrained_model():
    """
    Loads the pretrained version of VGG with the last layer cut off
    :return: pre-trained headless VGG16 Keras Model
    """
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    return model


def generate_features(numpy_arrays, model):
    """
    Takes in an array of image paths, and a trained model.
    Returns the activations of the last layer for each image
    :param image_paths: array of image paths
    :param model: pre-trained model
    :return: array of last-layer activations, and mapping from array_index to file_path
    """
    start = time.time()
    images = np.zeros(shape=(len(numpy_arrays), 224, 224, 3))
    file_mapping = {i: f for i, f in enumerate(numpy_arrays)}

    # We load all our dataset in memory because it is relatively small
    for i, img in enumerate(numpy_arrays):
        # img = image.load_img(f, target_size=(224, 224))

        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    logger.info("%s images loaded" % len(images))
    inputs = preprocess_input(images)
    logger.info("Images preprocessed")
    images_features = model.predict(inputs)
    end = time.time()
    logger.info("Inference done, %s Generation time" % (end - start))
    return images_features, file_mapping

