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

### Data preparation:

## Training the model

### Running training loop

### Inspecting training results

## Testing the model
