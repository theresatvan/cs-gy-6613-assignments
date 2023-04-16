# Sports Analytics - Object Tracking and Kalman Filters

## Task 1: Deep-SORT (40 points)

Diagram of Deep-SORT algorithm:

![Deep SORT diagram](images/deep_sort.drawio.png)

### Key Components:
1. *Object Detection:* This module is in charge of detecting objects in each frame of the input video sequence and create bounding boxes for each object in the frame. In our case, we use a pre-trained Faster R-CNN model, PyTorch's `fasterrcnn_resnet50_fpn_v2`. 

2. *Deep Appearance Descriptor:* The Object Detection module then sends its output as inputs to the Deep Appearance Descriptor. The Deep Appearance Descriptor module is part of the Data Association phase and is a convolutional neural network that extracts feature vectors for detected objects. These feature vectors are not affected by occlusion, lighting, or positioning.

3. *Hungarian algorithm:* The Hungarian algorithm is used to compute the optimal assignment of detections across frames in the Data Association phase of the Deep SORT algorithm. It uses an association metric that measures bounding box overlap.

4. *Matching Cascade:* In order to mitigate the issue of occlusion in object tracking, the Matching Cascade step solves the assignment problem in a seres of subproblems rather than using a global assignment approach. The longer an object is occluded, the more uncertain subsequent Kalman filter predictions associated with the object location are. Matching cascade gives priority to more frequently seen objects.

5. *Tracking Management:* When objects enter or leave the image, tracking IDs must be created or destroyed accordingly. Tracking Management phase either outputs the final video sequence with the object bounding boxes and tracking IDs or outputs to the Kalman Filter stage.

6. *Kalman Filter:* 

## Task 2: Deep-SORT Implementation (50 points)

This repository contains code that implements Deep-SORT with PyTorch's Faster R-CNN models for object detection. 

Install the requirements:
```
pip install -r requirements.txt
```

The input video is `Football_match.mp4`. 

In order to run the program:
```
python demo.py
```

The output will be located at `output/output.mp4`.

### References:

- Deep-SORT repository: https://github.com/nwojke/deep_sort
- Deep-SORT with YOLOv3 repository: https://github.com/anushkadhiman/ObjectTracking-DeepSORT-YOLOv3-TF2
- DeepSORT with YOLOv3 Kaggle notebook: https://www.kaggle.com/code/sakshaymahna/deepsort/notebook
- Faster R-CNN Object Detection with PyTorch: https://learnopencv.com/faster-r-cnn-object-detection-with-pytorch/

## Task 3: Critique (10 points)

When an object is occluded, the Deep-SORT algorithm will fail to assign the object its original tracking ID in subsequent frames. You can notice this with our output video especially with the sports ball. Everytime the sports ball is occluded, you can see that a different ID will be assigned to it in later frames when it reappears. This is also noticeable with several players in the video as well. 