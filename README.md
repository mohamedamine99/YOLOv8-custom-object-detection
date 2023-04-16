# YOLOv8 custom object detection

Welcome to my GitHub repository for custom object detection using [YOLOv8](https://github.com/ultralytics/ultralytics) by [Ultralytics](https://ultralytics.com/)! This project covers a range of object detection tasks and techniques, including utilizing a pretrained YOLOv8-based network model for [PPE object detection](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8), training a custom YOLOv8 model to recognize a single class (in this case, alpacas), and developing a multiclass object detector to recognize ants and insects.

To make this project accessible to all, I have leveraged Google Colab and Kaggle, providing easy-to-follow code and instructions for each stage of the project. Additionally, I have integrated my previously developed module `yolo_detect_and_count.py` for object detection, tracking, and counting with YOLOv8, streamlining the object detection process for various applications.

Whether you're an experienced data scientist or just starting with computer vision, this repository provides valuable insights into the world of custom object detection using YOLOv8.

## Navigating this repository


- [Custom-object-detection-with-YOLOv8](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8):  Directory for training and testing custom object detection models basd on YOLOv8 architecture, it contains the following folders files:
  - [class-descriptions-boxable.csv](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/class-descriptions-boxable.csv) : a csv file that contains all the IDs coreesponding to the classes provided by the OpenImages library for objecr detection
  - **Alpaca detector** (single class object detector): 
    - [-Kaggle- Alpaca detector-train.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/-Kaggle-%20Alpaca%20detector-train.ipynb): Kaggle notebook that demonstrates how to preprocess data to train a single class (alpaca) object detector based on YOLOv8 architecture.
    - [alpaca training results](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/alpaca%20training%20results): a folder that contains training results data for the alpaca detector.
    - [config -colab-.yaml](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/config%20-colab-.yaml) and
    [config -kaggle-.yaml](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/config%20-kaggle-.yaml): YAML files for training configuration for both Kaggle and colab versions.
  - **Ant and Insect classes detector** ( multi-class object detector):
    - [-Kaggle- Ant and Insect detector-train.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/-Kaggle-%20Ant%20and%20Insect%20detector-train.ipynb):  Kaggle notebook that demonstrates how to preprocess data to train a multi-class (Ant class and Insect class) object detector based on YOLOv8 architecture.
    - [Ant and insect training results](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results):  a folder that contains training results data for the And and Insect detector.
  
  
  
- [PPE-cutom-object-detection-with-YOLOv8](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8): Directory for personal protective equipement detection, it contains the following folders files:
  - [YOLOv8_PPE_object_detection.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/YOLOv8_PPE_object_detection.ipynb): google colab notebook for PPE object detection.
  - [ppe.pt](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/ppe.pt): PPE detection model, pre-trained.
  - [test imgs](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8/test%20imgs) and [img results](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8/img%20results): folders that contain testing images and resulting images with annotated PPE information.
  - [yolo_detect_and_count.py](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/yolo_detect_and_count.py) : python module, developed i a previous project that provides simple classes for object detection and object tracking and counting with YOLOv8.
  - [requirements.txt](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/requirements.txt) requirements for the [sort.py](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/sort.py) which itself is used by the `yolo_detect_and_count.py` module.
  
  
## Data Collection and Preprocessing for Object Detection using YOLOv8

In this section we will go through all the steps necessary to collect and preprocess data in order to prepare it to be trained for object detection using YOLOv8.

### Data collection:

To collect diverse and representative data for object detection using YOLOv8, or generally any other object detection model, the [Open Images](https://storage.googleapis.com/openimages/web/visualizer/index.html) library provides a valuable resource that includes millions of well-labeled images with a wide range of object classes.  
For more details about how to download and understand data provided by this library chech the following [link](https://storage.googleapis.com/openimages/web/download_v7.html).  
For the rest of this data collection section, all data will be downloaded programatically (in script, no need for manual download).

#### Downloading annotations and metadata for training, validation and (optional) testing

Let's first start by downloading training, validation and testing annotations and metadata.

```py
# training annotations and metadata
!wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv # 2.1G

# validation annotations and metadata
!wget https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv # 21M

# testing annotations and metadata
!wget https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv # 71M

```
`oidv6-train-annotations-bbox.csv` , `validation-annotations-bbox.csv` and `test-annotations-bbox.csv` are csv files that have training, validation and test metadata. All these files follow the same format. 
For more details check the [Open Images Dataset formats](https://storage.googleapis.com/openimages/web/download_v7.html#data-formats).

```
ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside,XClick1X,XClick2X,XClick3X,XClick4X,XClick1Y,XClick2Y,XClick3Y,XClick4Y
```
**P.S** : At this point we have only downloaded the metadata CSV files and not the actual image files. The reason for this is that we only need a specific subset of the Open Images dataset for our target objects, and downloading the entire dataset of 1.9 million images would be both time-consuming and unnecessary. 
By downloading only the necessary metadata files and selecting a subset of the images, we can save time and storage space while still obtaining high-quality data for our YOLOv8 model.

#### Selecting a Sub-Dataset for Object Detection: Choosing the Right Data for Your YOLOv8 Model
For more dtails about this important part of data collection check the [Open Images Download Section](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually)  

This section will explain the main strategy behind building a sub-dataset, with image data, for specific objects we want our model to detect.  
**We will simply follow the Open Image guidelines**. The main approach at this point is to create a text file, `image_list_file.txt` containing all the image IDs that we're interested in downloading. These IDs  come from filtering the annotations with certain classes. The text file must follow the following format
 : `$SPLIT/$IMAGE_ID`, where `$SPLIT` is either "train", "test", "validation"; and `$IMAGE_ID` is the image ID that uniquely identifies the image. A sample file could be:
```
train/f9e0434389a1d4dd
train/1a007563ebc18664
test/ea8bfd4e765304db
```
Now let's get to the part where we actually download images. Open Images provided a Python script that downloads images indicated by the `image_list_file.txt` we just created. First we download `downloader.py` by executing the following command :

```
!wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
```

Then we start the actual download :
```py
# run the donwloader script in order to download data related to the target objects 
# and according to the image_list_file.txt
!python downloader.py ./image_list_file.txt --download_folder=./data_all --num_processes=5
```


### Data preparation:

## Training the model

### Running training loop

### Inspecting training results

## Testing the model
