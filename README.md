# Deep Images Hub

This project create a platform to automatically verifying, preprocessing,and orangize users streaming images, and allow user to train image models on demand.

#### Motivation

The demands for machine learning on mobile and IoT devices are getting higher and higher especially for computer vision applications. Image classification and object detection algorithms are getting more mature and efficient to run on mobile devices with expected accuracy (ex. MobileNet V2). Companies like Google and Apple are rolling out DL/ML frameworks like Tensorflow Mobile and CoreML. However, there are still a gap to meet the needs from end users. There are not sufficient datasets that are dedicated for mobile image classification, and models that meets user’s specific needs. Deep iamges Hub is a solution to attempt to solve this problem by creating a platform to allow users to submit labeled images to the platfrom to crowsourcing imagedataset with different labels. The platform will automatically verify the image if it is matching with the labels. And it will assign the image to proper fodler where it belongs according to it's label. On the other hand, user who wants to train their customized computer vision model, can select the existing lables that are avilable on the platfrom, or list a new request for new labels so the image suppliers will know what object to collect and submit. Customized model will be train according to user's preferences.

[Slides](https://docs.google.com/presentation/d/17XCa3oY8J-khs3DmT14Esi0rPLR4x-ynFPMEQ80cagw/edit#slide=id.g36132c4481_0_39)

<hr/>

#### Problems to be Solved

Where are the Datasets? 
Do we have enough models to meet the demands from consumers?
Are we effective enough to train our model for specific needs / domains ?

Real-world deep learning applications are complex big data pipelines, which require a lot of data processing (such as cleaning, transformation, augmentation, feature extraction, etc.) beyond model training/inference. Therefore, it is much simpler and more efficient (for development and workflow management) to seamlessly integrate deep learning functionalities into existing big data workflow running on the same infrastructure, especially given the recent improvements that reduce deep learning training time from weeks to hours or even minutes.

Deep learning is increasingly adopted by the big data and data science community. Unfortunately, mainstream data engineers and data scientists are usually not deep learning experts; as the usages of deep learning expand and scale to larger deployment, it will be much more easier if these users can continue the use of familiar software tools and programming models (e.g., Spark or even SQL) and existing big data cluster infrastructures to build their deep learning applications.


#### Data

* Open Images subset with Image-Level Labels (19,995 classes & 5,655,108 images).
* Dataset will be loaded locally onto the nodes used for ingestion.

<hr/>

#### Model

* Model is pre-trained ResNet_v1(num_classes=5000).

<hr/>

#### Pipeline

<hr/>

#### Data Flow



<hr/>

#### Setup

* Initially, Spark/TensorflowOnSpark was run on one 4-node cluster.


<hr/>

#### Execution



<hr/>

#### Challenges

1. Need to prepare the Open Images Dataset and group them into Image-Level Labels first.
2. Create a pipeline to simulate data transferring between mobile to cloud distributed system at scale.
3. Automate filtering, preprocessing and data organizing on the fly. 
4. Automatically verifying and clustering images uploaded by users.


###### Performance Optimizations:
