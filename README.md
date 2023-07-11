# YOLOv8 custom object detection

<p align="center">
    <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/GIFs/bee.gif" width=350>
    <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/GIFs/Alpaca.gif" width=350>
    <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/GIFs/butterfly.gif" width=350>
    
    
</p>

Welcome to my GitHub repository for custom object detection using [YOLOv8](https://github.com/ultralytics/ultralytics) by [Ultralytics](https://ultralytics.com/)! 

This project covers a range of object detection tasks and techniques, including utilizing a pre-trained YOLOv8-based network model for [PPE object detection](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8), training a custom YOLOv8 model to recognize a single class (in this case, alpacas), and developing multiclass object detectors to recognize bees and butterflies, ants and insects.

To make this project accessible to all, I have leveraged Google Colab and Kaggle, providing easy-to-follow code and instructions for each stage of the project. Additionally, I have integrated my previously developed module `yolo_detect_and_count.py` for object detection, tracking, and counting with YOLOv8, streamlining the object detection process for various applications.

Whether you're an experienced data scientist or just starting with computer vision, this repository provides valuable insights into the world of custom object detection using YOLOv8.

The training process is automated for efficient training of custom object detection models. Simply specify detectable classes and training hyperparameters, and the code will take care of the rest, including downloading proper datasets, reorganizing the dataset in the YOLO-compatible format, generating proper YAML files, starting the training process, and automatically saving the results.


## Navigating this repository


- [Custom-object-detection-with-YOLOv8](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8):  Directory for training and testing custom object detection models basd on YOLOv8 architecture, it contains the following folders files:
  - [YOLOv8_Custom_Object_detector.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/YOLOv8_Custom_Object_detector.ipynb): an implementation example for the trained models.
  - [class-descriptions-boxable.csv](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/class-descriptions-boxable.csv) : a CSV file that contains all the IDs corresponding to the classes provided by the OpenImages library for object detection
  - **Alpaca detector** (single class object detector): 
    - [-Kaggle- Alpaca detector-train.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/-Kaggle-%20Alpaca%20detector-train.ipynb): Kaggle notebook that demonstrates how to preprocess data to train a single class (alpaca) object detector based on YOLOv8 architecture.
    - [alpaca training results](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/alpaca%20training%20results): a folder that contains training results data for the alpaca detector.
    - [config -colab-.yaml](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/config%20-colab-.yaml) and
    [config -kaggle-.yaml](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/config%20-kaggle-.yaml): YAML files for training configuration for both Kaggle and colab versions.
    
  - **Bee and Butterfly classes detector** ( multi-class object detector):
    - [Bees and Butterflies YOLOv8_Custom_Object_detector.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/Bees%20and%20Butterflies%20YOLOv8_Custom_Object_detector.ipynb):  Kaggle notebook that demonstrates how to preprocess data to train a multi-class (Bee class and Butterfly class) object detector based on YOLOv8 architecture.
    - [Bee and Butterfly 60 epochs](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/Bee%20and%20Butterfly%2060%20epochs):  a folder that contains training results data for the Bee and Butterfly detector for **60 epochs** of training.
    
    
  - **Ant and Insect classes detector** ( multi-class object detector):
    - [-Kaggle- Ant and Insect detector-train.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/-Kaggle-%20Ant%20and%20Insect%20detector-train.ipynb):  Kaggle notebook that demonstrates how to preprocess data to train a multi-class (Ant class and Insect class) object detector based on YOLOv8 architecture.
    - [Ant and insect training results 5 epochs](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%205%20epochs):  a folder that contains training results data for the And and Insect detector for **5 epochs**.
    - [Ant and insect training results 45 epochs](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%2045%20epochs):  a folder that contains training results data for the And and Insect detector for **45 epochs**.
  
  
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
- For more details about how to download and understand data provided by this library chech the following [link](https://storage.googleapis.com/openimages/web/download_v7.html).  
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
- For more dtails about this important part of data collection check the [Open Images Download Section](https://storage.googleapis.com/openimages/web/download_v7.html#download-manually)  

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
# run the downloader script in order to download data related to the target objects 
# and according to the image_list_file.txt
!python downloader.py ./image_list_file.txt --download_folder=./data_all --num_processes=5
```

### Data preparation:

After fully downloading our data, it is crucial to convert it into a YOLO-compatible format.
The format of a YOLO dataset consists of one text file per image, where each line in the text file contains the label and coordinates of an object in the image. The label and coordinates are separated by spaces, and the coordinates are normalized by the width and height of the image. The format of each line is as follows:

```csharp
<object-class> <x> <y> <width> <height>
```
where `<object-class>` is the name of the object class (0, 1, ...) , `<x>` and `<y>` are the center coordinates of the bounding box, and `<width>` and `<height>` are the width and height of the bounding box, respectively.

These files are organized in the following structure:
- A folder named `images` containing folders for each set : `training`, `validation` and `testing` containing all the images used for training validation and testing the YOLO model.
- A folder named `labels` containing  folders for each set : `training`, `validation` and `testing`, containing the corresponding label files for each image. Each label file should have the same name as its corresponding image file and be in the YOLO format.

```
├── images/
│   ├── training/
│   │   ├── imageID_1.jpg
│   │   ├── imageID_2.jpg
│   │   ├── ...
│   │   └── imageID_n.jpg
│   ├── validation/
│   │   ├── imageID_1.jpg
│   │   ├── imageID_2.jpg
│   │   ├── ...
│   │   └── imageID_m.jpg
│   └── testing/
│       ├── imageID_1.jpg
│       ├── imageID_2.jpg
│       ├── ...
│       └── imageID_k.jpg
└── labels/
    ├── training/
    │   ├── imageID_1.txt
    │   ├── imageID_2.txt
    │   ├── ...
    │   └── imageID_n.txt
    ├── validation/
    │   ├── imageID_1.txt
    │   ├── imageID_2.txt
    │   ├── ...
    │   └── imageID_m.txt
    └── testing/
        ├── imageID_1.txt
        ├── imageID_2.txt
        ├── ...
        └── imageID_k.txt

```

## Training the model

Now that our dataset is ready let's get to the training part
### Preparing the configuration YAML file
In order to train a YOLOv8 model for object detection, we need to provide specific configurations such as the dataset path, classes and training and validation sets. These configurations are typically stored in a YAML (Yet Another Markup Language) file which serves as a single source of truth for the model training process. This allows for easy modification and replication of the training process, as well as providing a convenient way to store and manage configuration settings.

This YAML file should follow this format:
```YAML
path: /kaggle/working/data  # Use absolute path 
train: images/train
val: images/validation

names:
  0: Ant
  1: Insect

```


### Running training loop
Running the training loop is very simple and thanks to ultralytics easy to use `train` method. for more details about model training for YOLOv8 check ultalytics [documentation](https://docs.ultralytics.com/modes/train)

```py
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  

# Use the model for training
results = model.train(data='/content/config.yaml', epochs=150)  # train the model

```

### Inspecting training results
The `train` method automatically saves the results in `./runs/detect/train`. These results include model weights (best.pt and last.pt), plots for various metrics (mAP50, mAP50-95, class loss, F1 score ,etc...) a quick visualization of some train and validation batches, a results.csv file that summarizes training results, etc..  
 
 This an example for Ant and Insect object detector trained for **only 5 epochs** with a pre-trained yolov8n model:
 
 <p align="center">
    <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%205%20epochs/results.png" width=600>
      <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%205%20epochs/val_batch2_pred.jpg" width=500>
</p>


 This an example for Ant and Insect object detector trained for **only 45 epochs** with a pre-trained yolov8n model:
 
 <p align="center">
    <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%2045%20epochs/results.png" width=600>
      <img src="https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/Custom-object-detection-with-YOLOv8/Ant%20and%20insect%20training%20results%20%2045%20epochs/val_batch2_pred.jpg" width=500>
</p>
