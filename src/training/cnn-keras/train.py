# USAGE
# python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pyimagesearch.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
from configparser import ConfigParser
import random
import pickle
import cv2
import os
import psycopg2
import datetime
from os.path import dirname as up



"""
Commonly Shared Statics

"""

# Set up project path
projectPath = os.getcwd()

s3_bucket_name = "insight-deep-images-hub"

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



def save_results_to_db(request_number,final_acc,final_val_acc,final_loss,final_val_loss,summary_note):



	sql = """

    UPDATE training_records
	SET final_accuracy = %s,
 	final_validation_accuracy = %s,
	final_loss = %s,
	final_validation_loss = %s,								  
	creation_date	= %s,
	note = %s			  
    WHERE model_id = %s;								  
					

    """


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

		# writing image info into the database
		# execute a statement
		print('writing image batch info into the database...')
		cur.execute(sql,(final_acc,final_val_acc,final_loss,final_val_loss,datetime.datetime.now(),summary_note,request_number))

		# commit the changes to the database
		conn.commit()

		# close the communication with the PostgreSQL
		cur.close()

	except (Exception, psycopg2.DatabaseError) as error:

		print(error)
	finally:
		if conn is not None:
			conn.close()
		print('Database connection closed.')




# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
				help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", required=True,
				help="path to output model")
ap.add_argument("-l", "--labelbin", required=True,
				help="path to output label binarizer")
ap.add_argument("-tn", "--training_request_number", required=True,
				help="this model training request number")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
				help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
EPOCHS = 5
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# initialize the data and labels
data = []
labels = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
print("[INFO] data matrix: {:.2f}MB".format(
	data.nbytes / (1024 * 1000.0)))

# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
												  labels, test_size=0.2, random_state=42)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
						 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
						 horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
							depth=IMAGE_DIMS[2], classes=len(lb.classes_))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,
			  metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)



final_acc = H.history['acc'][-1]
final_val_acc = H.history['val_acc'][-1]
final_loss = H.history['loss'][-1]
final_val_loss = H.history['val_loss'][-1]


# print results
print("[INFO] Results:")
print("Final Accuracy: ", final_acc)
print("Final Validation Accuracy: ", final_val_acc)
print("Final Loss: ", final_loss)
print("Final Validation Loss: ", final_val_loss)


# save results to DB

summary_note = ""

if final_acc < 0.85:
	summary_note = "Final accuracy is below threshold 0.85. You may consider to re-train the model with more data or increase the epoch number"
else:
	summary_note = "Final accuracy is above threshold 0.85. This model can be a good baseline model"


save_results_to_db(args["training_request_number"],final_acc,final_val_acc,final_loss,final_val_loss,summary_note)



# save the model to disk
print("[INFO] serializing network...")

# create directory if not exist
if not os.path.exists('/tmp/Deep_image_hub_Model_Training/model/'):
	os.makedirs('/tmp/Deep_image_hub_Model_Training/model/')

model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


# delete tmp dataset
import shutil
shutil.rmtree("/tmp/Deep_image_hub_Model_Training/dataset")