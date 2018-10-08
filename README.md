
![Alt text](README_images/Deep_image_hub_logo.png?raw=true "Optional Title")


-----------------


# Deep Images Hub

This project create a platform to automatically verify, preprocess and orangize cloudsourced labeled images as datasets for users to select and download by their choice. Additionally, the platform allow premium users to train a sample computer vision model with their choice of image subsets. 


#### Motivation

![Alt text](README_images/MealPic_App_at_NYC_Media_Lab_Summit_2017_Demo%20(1).gif?raw=true "MealPic App at NYC Media Lab Summit 2017 Demo")

![Alt text](README_images/Motivation_1.png?raw=true "Motivation")

Deep Images Hub. It is a platform that inspired by my previous venture MealPic. MealPic was a startup to try to use computer vision, augmented reality and personalized recommendation to help pregnant women to identify packaged food that meet both their craving and personal health. During our customer discovery and market researching, we discovered that consumer products and services use cases for using computer vision to recognize objects on mobile and IoT is on great demands. Even Amazon is trying to tap into this market by pushing their AWS Deeplens with their cloud services.


However, in order to train a computer vision model that can correctly recognize the object under all kind of condition of lighting and background environment, we need a large volume of diversified images for a single object. So for computer vision startups like MealPic we just couldn't afford the resources to collect all kind of images for different objects.

For example, we need many Barack Obama's images to reconize him!
![Alt text](README_images/Example%20for%20reason%20of%20diversified%20images.jpg?raw=true "For example we need many Barack Obama's images to reconize him")


So! I want to create a platform that can artificial the artificial intelligence training process just like Amazon’s Mechanical Turk, I am here to introduce the Deep Images Hub

![Alt text](README_images/Simple%20Platform%20Blueprint.png?raw=true "Motivation")


More details can be found in this more indepth slides:
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
