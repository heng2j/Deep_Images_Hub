
![Alt text](README_images/Deep_image_hub_logo_new.png?raw=true "Optional Title")


-----------------


# Deep Images Hub

This project create a platform to automatically verify, preprocess and orangize crowdsourced labeled images as datasets for users to select and download by their choice. Additionally, the platform allow premium users to train a sample computer vision model with their choice of image subsets. 


#### Motivation

![Alt text](README_images/MealPic_App_at_NYC_Media_Lab_Summit_2017_Demo%20(1).gif?raw=true "MealPic App at NYC Media Lab Summit 2017 Demo")

![Alt text](README_images/Motivation_1.png?raw=true "Motivation")

Deep Images Hub. It is a platform that inspired by my previous venture MealPic. MealPic was a startup to try to use computer vision, augmented reality and personalized recommendation to help pregnant women to identify packaged food that meet both their craving and personal health. During our customer discovery and market researching, we discovered that consumer products and services use cases for using computer vision to recognize objects on mobile and IoT is on great demands. Even Amazon is trying to tap into this market by pushing their AWS Deeplens with their cloud services.


However, in order to train a computer vision model that can correctly recognize the object under all kind of condition of lighting and background environment, we need a large volume of diversified images for a single object. So for computer vision startups like MealPic we just couldn't afford the resources to collect all kind of images for different objects.

For example, we need many Barack Obama's images to reconize him!
![Alt text](README_images/Example%20for%20reason%20of%20diversified%20images.jpg?raw=true "For example we need many Barack Obama's images to reconize him")


So! I want to create a platform that can artificial the artificial intelligence training process just like Amazon’s Mechanical Turk, and here I am introducing the Deep Images Hub

![Alt text](README_images/Simple%20Platform%20Blueprint.png?raw=true "Motivation")


More details can be found in this more indepth slides:
[Slides](https://docs.google.com/presentation/d/17XCa3oY8J-khs3DmT14Esi0rPLR4x-ynFPMEQ80cagw/edit#slide=id.g36132c4481_0_39)

<hr/>

#### Painpoints to be Solved

There are not enough diversified image datasets and computer vision model to meet the demands from different consumer products and services use cases?

There is not enough resource to collect large volume of diversified images for a single object. Especially for low budgeted startups.

Real-world deep learning applications are complex big data pipelines, which require a lot of data processing (such as cleaning, transformation, augmentation, feature extraction, etc.) beyond model training/inference. Therefore, it is much simpler and more efficient (for development and workflow management) to seamlessly integrate deep learning functionalities into existing big data workflow running on the same infrastructure, especially given the recent improvements that reduce deep learning training time from weeks to hours or even minutes.

Deep learning is increasingly adopted by the big data and data science community. Unfortunately, mainstream data engineers and data scientists are usually not deep learning experts; as the usages of deep learning expand and scale to larger deployment, it will be much more easier if these users can continue the use of familiar software tools and programming models (e.g., Spark or even SQL) and existing big data cluster infrastructures to build their deep learning applications.

<hr/>

#### Data

* Open Images subset with Image-Level Labels (1.74 million images and 600 classes with hierarchies). [Source](https://storage.googleapis.com/openimages/web/factsfigures.html)

* Dataset are stored in image source S3 bucket. An automated process will be copy the images from the source S3 bucket to Image Data Hub's bucket with preprocessings to simulate cloudsourced labled imges submissions. 

<hr/>

#### Pipeline

![Alt text](README_images/Pipeline.png?raw=true "For Future Distributed Ready Training Pipeline")


<hr/>

#### Database Schema

![Alt text](README_images/Database%20Schema.png?raw=true "Databae Schema")


<hr/>

#### Model

* Model is pre-trained VGG16(num_classes=1000).

<hr/>



#### Data Flow



<hr/>

#### Setup

* Initially, Spark/TensorflowOnSpark was run on one 4-node cluster.


<hr/>

#### Execution

Please check Deep Images Hub [wiki site](../../wiki) for detailed
documentations. Here are the table of contents of how to get start:

* [Data Prepartion](../../wiki/Data-Preparation) 
  * Can also reference the Juypter Notebook for [Exploratory Data Analysis on the Pen Images classes](https://github.com/heng2j/Deep_Images_Hub/blob/master/doc/noteBooks/Exploratory%20Data%20Analysis%20on%20Open%20Images%20Classes.ipynb))
* [Backend Design and Implementtation](../../wiki/Design-and-Planing)
  * [Backend Design and Implementtation](/sql)
* [Simulating Crowdsourcing Images Workflow](../../wiki/Image-Suppliers-Implementation)
* [Business User Requests on Deep Image Hub Website Workflow](../../wiki/Business-Users-Requests-Implementation)
* [Computer Vision Model Training Workflow](../../wiki/Premium-Business-Users-Requests-Implementation)
* [Summary & Future Works](../../wiki/Results-and-Future-Works)



<hr/>

#### Challenges

1. How to simulate crowdsourced labeled image submission?
2. How can we preprocess the crowdsourced images in our centralized data hub?
3. How to design a platfrom that host image datasets that are ready for users to download and train with distributedly on demands?
4. How to train deep learning model distributed with centralized data source?
5. How to work with deep learning in Spark when it is still in it's infancy.
6. Work flows to connect all the dots for the platform


###### Performance Optimizations:
