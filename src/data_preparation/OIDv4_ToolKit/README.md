**Tools to download Images from Open Images:**

Please use [OIDv4_ToolKit](https://github.com/EscVM/OIDv4_ToolKit) to download the images

The ToolKit can only down one label at a time, I wrote this [OIDv4_batch_image_downloader] bash script to download all 600 labels by looping through a text file that contain call the labels. Since the script will automatically remove the labels that had been downloaded from the text file. So if the download process failed or took too much time it can be rerun.




* Dataset are stored in image source S3 bucket. And I have this simple script to help me to copy the Images from OIDv4_ToolKit/OID/Dataset/ to my s3://insight-data-images/dataset/