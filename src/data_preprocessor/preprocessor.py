#!/usr/bin/env python2
# preprocessor.py
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


import logging
from keras_preprocessing import image
import time
import os
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from os.path import dirname as up


"""
Commonly Shared Statics

"""

# Set up project path
projectPath = up(up(os.getcwd()))

s3_bucket_name = "s3://insight-data-images/"

database_ini_file_path = "/utilities/database/database.ini"





logger = logging.getLogger()
logger.setLevel(logging.INFO)

def to_array(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x_raw = image.img_to_array(img)
    return x_raw.astype(np.uint8)


def load_paired_img_wrd(folder):
    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []
    for cl in class_names:
        subfiles = [f for f in os.listdir(folder + "/" + cl) if ".DS" not in f]
        for subf in subfiles:
            full_path = os.path.join(folder, cl, subf)
            img = image.load_img(full_path, target_size=(224, 224))
            x_raw = image.img_to_array(img)
            x_expand = np.expand_dims(x_raw, axis=0)
            x = preprocess_input(x_expand)
            image_list.append(x)
            paths_list.append(full_path)
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, np.array(labels_list), paths_list

def load_images_by_label(folder,label):

    class_names = [fold for fold in os.listdir(folder) if ".DS" not in fold]
    image_list = []
    labels_list = []
    paths_list = []

    subfiles = [f for f in os.listdir(folder + "/" + label) if ".DS" not in f]
    for subf in subfiles:
        full_path = os.path.join(folder, label, subf)
        img = image.load_img(full_path, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        x = preprocess_input(x_expand)
        image_list.append(x)
        paths_list.append(full_path)
    img_data = np.array(image_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]

    return img_data, np.array(labels_list), paths_list

def load_headless_pretrained_model():
    """
    Loads the pretrained version of VGG with the last layer cut off
    :return: pre-trained headless VGG16 Keras Model
    """
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=pretrained_vgg16.input,
                  outputs=pretrained_vgg16.get_layer('fc2').output)
    return model


def generate_features(image_paths, model):
    """
    Takes in an array of image paths, and a trained model.
    Returns the activations of the last layer for each image
    :param image_paths: array of image paths
    :param model: pre-trained model
    :return: array of last-layer activations, and mapping from array_index to file_path
    """
    start = time.time()
    images = np.zeros(shape=(len(image_paths), 224, 224, 3))
    file_mapping = {i: f for i, f in enumerate(image_paths)}

    # We load all our dataset in memory because it is relatively small
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size=(224, 224))

        print("f type: ", f)
        print("img type: ", type(img))
        print(img)

        x_raw = image.img_to_array(img)

        print("type of x_raw: ",type(x_raw) )

        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    logger.info("%s images loaded" % len(images))
    inputs = preprocess_input(images)
    logger.info("Images preprocessed")
    images_features = model.predict(inputs)
    end = time.time()
    logger.info("Inference done, %s Generation time" % (end - start))
    return images_features, file_mapping


if __name__ == '__main__':

    from scipy import spatial

    images, vectors, image_paths = load_images_by_label(projectPath+'/' +'data/images/dummy_dataset_1000','diningtable')

    model = load_headless_pretrained_model()

    images_features, file_index = generate_features(image_paths,model)

    print(image_paths)
    print(vectors)
    print(images_features[0])
    print(images_features[1])
    print(file_index)



    # bird
    # ['dataset/bird/2008_008490.jpg', 'dataset/bird/2008_004452.jpg', 'dataset/bird/2008_007205.jpg', 'dataset/bird/2008_007003.jpg', 'dataset/bird/2008_005774.jpg', 'dataset/bird/2008_001673.jpg', 'dataset/bird/2008_006281.jpg', 'dataset/bird/2008_002970.jpg', 'dataset/bird/2008_004087.jpg', 'dataset/bird/2008_008194.jpg', 'dataset/bird/2008_008354.jpg', 'dataset/bird/2008_003997.jpg', 'dataset/bird/2008_006186.jpg', 'dataset/bird/2008_003160.jpg', 'dataset/bird/2008_004551.jpg', 'dataset/bird/2008_008347.jpg', 'dataset/bird/2008_005924.jpg', 'dataset/bird/2008_007854.jpg', 'dataset/bird/2008_004805.jpg', 'dataset/bird/2008_007317.jpg', 'dataset/bird/2008_000095.jpg', 'dataset/bird/2008_004783.jpg', 'dataset/bird/2008_004973.jpg', 'dataset/bird/2008_008185.jpg', 'dataset/bird/2008_007498.jpg', 'dataset/bird/2008_008002.jpg', 'dataset/bird/2008_003222.jpg', 'dataset/bird/2008_004362.jpg', 'dataset/bird/2008_002471.jpg', 'dataset/bird/2008_008376.jpg', 'dataset/bird/2008_008404.jpg', 'dataset/bird/2008_001829.jpg', 'dataset/bird/2008_001020.jpg', 'dataset/bird/2008_003232.jpg', 'dataset/bird/2008_005279.jpg', 'dataset/bird/2008_003580.jpg', 'dataset/bird/2008_007839.jpg', 'dataset/bird/2008_006924.jpg', 'dataset/bird/2008_005208.jpg', 'dataset/bird/2008_003484.jpg', 'dataset/bird/2008_007948.jpg', 'dataset/bird/2008_006667.jpg', 'dataset/bird/2008_004689.jpg', 'dataset/bird/2008_005757.jpg', 'dataset/bird/2008_003087.jpg', 'dataset/bird/2008_002399.jpg', 'dataset/bird/2008_007587.jpg', 'dataset/bird/2008_005186.jpg', 'dataset/bird/2008_007752.jpg', 'dataset/bird/2008_008461.jpg']


    # bicycle
    # ['dataset/bicycle/2008_007993.jpg', 'dataset/bicycle/2008_008097.jpg', 'dataset/bicycle/2008_005175.jpg', 'dataset/bicycle/2008_008725.jpg', 'dataset/bicycle/2008_008718.jpg', 'dataset/bicycle/2008_008320.jpg', 'dataset/bicycle/2008_004654.jpg', 'dataset/bicycle/2008_008708.jpg', 'dataset/bicycle/2008_006254.jpg', 'dataset/bicycle/2008_003072.jpg', 'dataset/bicycle/2008_008131.jpg', 'dataset/bicycle/2008_004441.jpg', 'dataset/bicycle/2008_004656.jpg', 'dataset/bicycle/2008_000090.jpg', 'dataset/bicycle/2008_008619.jpg', 'dataset/bicycle/2008_008368.jpg', 'dataset/bicycle/2008_001402.jpg', 'dataset/bicycle/2008_002679.jpg', 'dataset/bicycle/2008_007067.jpg', 'dataset/bicycle/2008_005276.jpg', 'dataset/bicycle/2008_008753.jpg', 'dataset/bicycle/2008_004592.jpg', 'dataset/bicycle/2008_007935.jpg', 'dataset/bicycle/2008_006154.jpg', 'dataset/bicycle/2008_007470.jpg', 'dataset/bicycle/2008_003819.jpg', 'dataset/bicycle/2008_003617.jpg', 'dataset/bicycle/2008_007103.jpg', 'dataset/bicycle/2008_006234.jpg', 'dataset/bicycle/2008_008755.jpg', 'dataset/bicycle/2008_000725.jpg', 'dataset/bicycle/2008_008572.jpg', 'dataset/bicycle/2008_004995.jpg', 'dataset/bicycle/2008_002894.jpg', 'dataset/bicycle/2008_003140.jpg', 'dataset/bicycle/2008_004363.jpg', 'dataset/bicycle/2008_001226.jpg', 'dataset/bicycle/2008_002129.jpg', 'dataset/bicycle/2008_001626.jpg', 'dataset/bicycle/2008_004603.jpg', 'dataset/bicycle/2008_003351.jpg', 'dataset/bicycle/2008_007719.jpg', 'dataset/bicycle/2008_006064.jpg', 'dataset/bicycle/2008_007421.jpg', 'dataset/bicycle/2008_001523.jpg', 'dataset/bicycle/2008_008671.jpg', 'dataset/bicycle/2008_004113.jpg', 'dataset/bicycle/2008_006467.jpg', 'dataset/bicycle/2008_008528.jpg', 'dataset/bicycle/2008_007222.jpg']

    # Cat
    # ['dataset/cat/2008_007039.jpg', 'dataset/cat/2008_000227.jpg', 'dataset/cat/2008_001885.jpg', 'dataset/cat/2008_007589.jpg', 'dataset/cat/2008_000345.jpg', 'dataset/cat/2008_005003.jpg', 'dataset/cat/2008_003063.jpg', 'dataset/cat/2008_005614.jpg', 'dataset/cat/2008_004873.jpg', 'dataset/cat/2008_007176.jpg', 'dataset/cat/2008_006081.jpg', 'dataset/cat/2008_000182.jpg', 'dataset/cat/2008_007363.jpg', 'dataset/cat/2008_006999.jpg', 'dataset/cat/2008_005857.jpg', 'dataset/cat/2008_006384.jpg', 'dataset/cat/2008_003607.jpg', 'dataset/cat/2008_002294.jpg', 'dataset/cat/2008_003772.jpg', 'dataset/cat/2008_005449.jpg', 'dataset/cat/2008_004347.jpg', 'dataset/cat/2008_007855.jpg', 'dataset/cat/2008_006194.jpg', 'dataset/cat/2008_004635.jpg', 'dataset/cat/2008_001836.jpg', 'dataset/cat/2008_006403.jpg', 'dataset/cat/2008_002329.jpg', 'dataset/cat/2008_007324.jpg', 'dataset/cat/2008_007496.jpg', 'dataset/cat/2008_005252.jpg', 'dataset/cat/2008_001592.jpg', 'dataset/cat/2008_003622.jpg', 'dataset/cat/2008_005469.jpg', 'dataset/cat/2008_002845.jpg', 'dataset/cat/2008_007888.jpg', 'dataset/cat/2008_000116.jpg', 'dataset/cat/2008_002067.jpg', 'dataset/cat/2008_000670.jpg', 'dataset/cat/2008_006956.jpg', 'dataset/cat/2008_004990.jpg', 'dataset/cat/2008_002410.jpg', 'dataset/cat/2008_004303.jpg', 'dataset/cat/2008_001290.jpg', 'dataset/cat/2008_005181.jpg', 'dataset/cat/2008_002201.jpg', 'dataset/cat/2008_004328.jpg', 'dataset/cat/2008_003244.jpg', 'dataset/cat/2008_001335.jpg', 'dataset/cat/2008_005386.jpg', 'dataset/cat/2008_002749.jpg']


    # Dog
    # ['dataset/dog/2008_005160.jpg', 'dataset/dog/2008_001676.jpg', 'dataset/dog/2008_001070.jpg', 'dataset/dog/2008_005798.jpg', 'dataset/dog/2008_004653.jpg', 'dataset/dog/2008_002395.jpg', 'dataset/dog/2008_006130.jpg', 'dataset/dog/2008_000620.jpg', 'dataset/dog/2008_005823.jpg', 'dataset/dog/2008_007567.jpg', 'dataset/dog/2008_003852.jpg', 'dataset/dog/2008_005563.jpg', 'dataset/dog/2008_001895.jpg', 'dataset/dog/2008_004044.jpg', 'dataset/dog/2008_005831.jpg', 'dataset/dog/2008_000053.jpg', 'dataset/dog/2008_004745.jpg', 'dataset/dog/2008_000641.jpg', 'dataset/dog/2008_005882.jpg', 'dataset/dog/2008_006356.jpg', 'dataset/dog/2008_002859.jpg', 'dataset/dog/2008_005065.jpg', 'dataset/dog/2008_002441.jpg', 'dataset/dog/2008_005890.jpg', 'dataset/dog/2008_002536.jpg', 'dataset/dog/2008_003576.jpg', 'dataset/dog/2008_007537.jpg', 'dataset/dog/2008_000270.jpg', 'dataset/dog/2008_005240.jpg', 'dataset/dog/2008_006602.jpg', 'dataset/dog/2008_007871.jpg', 'dataset/dog/2008_007694.jpg', 'dataset/dog/2008_000138.jpg', 'dataset/dog/2008_000706.jpg', 'dataset/dog/2008_007519.jpg', 'dataset/dog/2008_000897.jpg', 'dataset/dog/2008_007478.jpg', 'dataset/dog/2008_004950.jpg', 'dataset/dog/2008_001220.jpg', 'dataset/dog/2008_005046.jpg', 'dataset/dog/2008_004833.jpg', 'dataset/dog/2008_004760.jpg', 'dataset/dog/2008_001285.jpg', 'dataset/dog/2008_004498.jpg', 'dataset/dog/2008_003133.jpg', 'dataset/dog/2008_004528.jpg', 'dataset/dog/2008_004931.jpg', 'dataset/dog/2008_006511.jpg', 'dataset/dog/2008_000808.jpg', 'dataset/dog/2008_001479.jpg']

    # Dining table
    # ['dataset/diningtable/2008_005570.jpg', 'dataset/diningtable/2008_007402.jpg', 'dataset/diningtable/2008_004321.jpg', 'dataset/diningtable/2008_003881.jpg', 'dataset/diningtable/2008_005348.jpg', 'dataset/diningtable/2008_004293.jpg', 'dataset/diningtable/2008_003477.jpg', 'dataset/diningtable/2008_000817.jpg', 'dataset/diningtable/2008_001077.jpg', 'dataset/diningtable/2008_000418.jpg', 'dataset/diningtable/2008_002384.jpg', 'dataset/diningtable/2008_006192.jpg', 'dataset/diningtable/2008_004948.jpg', 'dataset/diningtable/2008_007048.jpg', 'dataset/diningtable/2008_006008.jpg', 'dataset/diningtable/2008_002079.jpg', 'dataset/diningtable/2008_001809.jpg', 'dataset/diningtable/2008_001758.jpg', 'dataset/diningtable/2008_006750.jpg', 'dataset/diningtable/2008_000043.jpg', 'dataset/diningtable/2008_001230.jpg', 'dataset/diningtable/2008_005081.jpg', 'dataset/diningtable/2008_000885.jpg', 'dataset/diningtable/2008_008363.jpg', 'dataset/diningtable/2008_008388.jpg', 'dataset/diningtable/2008_007291.jpg', 'dataset/diningtable/2008_003753.jpg', 'dataset/diningtable/2008_004216.jpg', 'dataset/diningtable/2008_004564.jpg', 'dataset/diningtable/2008_008362.jpg', 'dataset/diningtable/2008_004776.jpg', 'dataset/diningtable/2008_006969.jpg', 'dataset/diningtable/2008_001155.jpg', 'dataset/diningtable/2008_007097.jpg', 'dataset/diningtable/2008_002892.jpg', 'dataset/diningtable/2008_003224.jpg', 'dataset/diningtable/2008_008365.jpg', 'dataset/diningtable/2008_004588.jpg', 'dataset/diningtable/2008_007692.jpg', 'dataset/diningtable/2008_002362.jpg', 'dataset/diningtable/2008_001723.jpg', 'dataset/diningtable/2008_005975.jpg', 'dataset/diningtable/2008_004289.jpg', 'dataset/diningtable/2008_002567.jpg', 'dataset/diningtable/2008_008098.jpg', 'dataset/diningtable/2008_001083.jpg', 'dataset/diningtable/2008_004851.jpg', 'dataset/diningtable/2008_008266.jpg', 'dataset/diningtable/2008_003534.jpg', 'dataset/diningtable/2008_001451.jpg']


    # model = load_headless_pretrained_model()
    #
    # images_features, file_index = generate_features(['dataset/dog/2008_001676.jpg','dataset/dog/2008_006130.jpg'],model)
    #
    # print(images_features[0])
    # print(images_features[1])
    #
    # Distance =  spatial.distance.cosine(images_features[0], images_features[1])
    #
    # print("Distance: ", Distance )


