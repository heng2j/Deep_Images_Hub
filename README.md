
![Alt text](README_images/Deep_image_hub_logo_new.png?raw=true "Optional Title")


-----------------


# Deep Images Hub

This project create a platform to automatically verify, preprocess and orangize crowdsourced labeled images as datasets for users to select and download by their choice. Additionally, the platform allow premium users to train a sample computer vision model with their choice of image subsets. 


[Deep Images Hub WebPage](http://www.deepimagehub.space:5000/)




#### Motivation

![Alt text](README_images/Motivation_2.png?raw=true "Motivation")

The performance of Computer Vision is now closest to human eyes thanks to the recent achievements from computer vision researchers. At the same time, the demands for Computer Vision is all time high. Especially for classifying and recognizing consumer products since it can create new business use cases and opportunities for IoT, mobile apps and Augmented Reality companies. But how come, computer Vision is still not a household name? 

The major bottleneck is lack of image dataset to train CV models.To train a good quality computer vision model required a lot of image data. For example if a we want to train a computer vision model to recognize this Lemon Flavor of LaCroix Sparkling Water. The data scientist will need approximately 1000+ images of the sparkling water under all kind of lighting and background environment. The problem is, where can the we find these many of images? Even the internet does have these many of images of the same product.


![Alt text](README_images/problem_statement.png?raw=true "Problem Statement")


Can’t imagine to train a model to recognize different soft drinks on the market. 

Without image datasets, model can’t be train and new business can’t be create. As a data engineer with a heart of an entrepreneur. I spotted this problem and try to come up with a solution with Deep Images Hub. 

Deep Images Hub is the centralized image data hub that provides crowdsourced labeled images from images suppliers to data scientist. Data Scientist can download the images by their choices of labels. 

Additionally for premium users, they can request to train a baseline model. So Deep Image Hub is the automated platform for computer vision. We envisoned to be the artificial the computer vison platform just like Amazon's Mechanical Turk.

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

#### Data Modeling and Database Schema

![Alt text](README_images/Database%20Schema.png?raw=true "Databae Schema")

 [All SQLs can be found here](/src/sql)

<hr/>

#### Model

* Model is pre-trained VGG16(num_classes=1000).

<hr/>



#### Data Flow



<hr/>

#### Setup

###### Prerequisites 
 * Please register for Amazon AWS account and set up your AWS CLI according to [Amazon’s documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-getting-started.html) 
 * Please install and configure Pegasus according to the [Github Instructions](https://github.com/InsightDataScience/pegasus)
 * Please create a PostgreSQL DB Instance with AWS RDS by following this [user guide](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_GettingStarted.CreatingConnecting.PostgreSQL.html)



<hr/>

#### Execution

Please check Deep Images Hub [wiki site](../../wiki) for detailed
documentations. Here are the table of contents of how to get start:

* [Data Prepartion](../../wiki/1.-Data-Preparation) 
  * Can also reference the Juypter Notebook for [Exploratory Data Analysis on the Pen Images classes](https://github.com/heng2j/Deep_Images_Hub/blob/master/src/noteBooks/Exploratory%20Data%20Analysis%20on%20Open%20Images%20Classes.ipynb))
* [Backend Design and Implementtation](../../wiki/2.-Design-and-Planing)
* [Simulating Crowdsourcing Images Workflow](../../wiki/3.-Image-Suppliers-Implementation)
* [Business User Requests on Deep Image Hub Website Workflow](../../wiki/4.-Business-Users-Requests-Implementation)
* [Computer Vision Model Training Workflow](../../wiki/5.-Premium-Business-Users-Requests-Implementation)
* [Summary & Future Works](../../wiki/6.-Results-and-Future-Works)



<hr/>

#### Challenges

1. How to simulate crowdsourced labeled image submission?
2. How can we preprocess the crowdsourced images in our centralized data hub?
3. How to design a platfrom that host image datasets that are ready for users to download and train with distributedly on demands?
4. How to train deep learning model distributed with centralized data source?
5. How to work with deep learning in Spark when it is still in it's infancy.
6. Work flows to connect all the dots for the platform


<hr/>

#### Libraries used
For geocoding and decoding:
[geopy](https://pypi.org/project/geopy/)



<hr/>

#### Extra References

These [references and resources](../../wiki/7.-References-&-Resources) I came across while I was working on this project. They gave me lights of how to solve some of the technically challenges.
