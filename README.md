# YOLOv8 custom object detection

Welcome to my GitHub repository for custom object detection using [YOLOv8](https://github.com/ultralytics/ultralytics) by [Ultralytics](https://ultralytics.com/)! This project covers a range of object detection tasks and techniques, including utilizing a pretrained YOLOv8-based network model for [PPE object detection](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8), training a custom YOLOv8 model to recognize a single class (in this case, alpacas), and developing a multiclass object detector to recognize ants and insects.

To make this project accessible to all, I have leveraged Google Colab and Kaggle, providing easy-to-follow code and instructions for each stage of the project. Additionally, I have integrated my previously developed module `yolo_detect_and_count.py` for object detection, tracking, and counting with YOLOv8, streamlining the object detection process for various applications.

Whether you're an experienced data scientist or just starting with computer vision, this repository provides valuable insights into the world of custom object detection using YOLOv8.

## Navigating this repository

- [PPE-cutom-object-detection-with-YOLOv8](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8): Directory for personal protective equipement detection, it contains the following folders files:
  - [YOLOv8_PPE_object_detection.ipynb](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/YOLOv8_PPE_object_detection.ipynb): google colab notebook for PPE object detection.
  - [ppe.pt](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/ppe.pt): PPE detection model, pre-trained.
  - [test imgs](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8/test%20imgs) and [img results](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/tree/main/PPE-cutom-object-detection-with-YOLOv8/img%20results): folders that contain testing images and resulting images with annotated PPE information.
  - [yolo_detect_and_count.py](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/yolo_detect_and_count.py) : python module, developed i a previous project that provides simple classes for object detection and object tracking and counting with YOLOv8.
  - [requirements.txt](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/requirements.txt) requirements for the [sort.py](https://github.com/mohamedamine99/YOLOv8-custom-object-detection/blob/main/PPE-cutom-object-detection-with-YOLOv8/sort.py) which itself is used by the `yolo_detect_and_count.py` module.
