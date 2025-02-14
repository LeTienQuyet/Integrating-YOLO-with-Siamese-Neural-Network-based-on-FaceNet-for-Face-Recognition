# Integrating YOLO with Siamese Neural Network based on FaceNet for Face Recognition
This repo utilizes **Siamese Neural Networks (SNNs) [1]** combined with **YOLO** to enhance the effectiveness of face recognition.
![pipeline](./image/pipeline.png)
## Members
|      Name             |     Gmail                |
| :---------------:     | :--------:               |
| *Tien Quyet Le*       | *21520428@gm.uit.edu.vn*   |
| *Cong Nguyen Nguyen*  | *21521200@gm.uit.edu.vn*   |
## Table of Contents
* [Table of Contents](#table-of-contents)
* [Main Modules](#main-modules)
    + [Detector](#detector)
    + [Backbone](#backbone)
* [Instruction](#instruction)
    + [Compare 2 Faces](#compare-2-faces)
    + [Football Player Recognition (To do)](#football-player-recognition-to-do)
    + [Training](#training)
* [References](#references)
## Main Modules
### Detector
Before inputting the image into the FaceNet model, we deploy a detector to identify the location most likely to contain a face, specifically the **YOLOv5n** model. This process allows us to accurately determine the area where a face may appear, thereby enhancing the accuracy and reliability of the model.\
After identifying the face location, we will use this region as input for the FaceNet model instead of the original image. The YOLOv5n model has been further trained on the **Face-Detection-Dataset [2]** for a total of 30 epochs to significantly improve its face detection capabilities.
### Backbone
The backbone used in the model is **Inception-ResNet (v1)**, pre-trained on the **VGGFace2** dataset. To retain the features learned from this extensive dataset, the entire backbone will remain frozen during training. The model will only update the parameters of the added Linear layer (has Dropout layer before).
```python
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(pretrained="vggface2")
```
## Instruction
Install required packages:
```
pip install -r requirements.txt
```
### Compare 2 Faces
To determine whether the two faces in the provided images belong to the same person, you can use the following command:
```
python compare2faces.py --img1_path /path/to/1st/image --img2_path /path/to/2nd/image
```
By default, the output will display **Same person !!!** or **Diff person !!!**. If you prefer to see the probability score instead of a definitive label, set `noti = False` in the command above.
![Example for compare](./image/example_compare.png)
### Football Player Recognition (To do)
### Training
Move to `src` and training by:
```
python train.py --num_epoch {epochs} \
                --lr {learning_rate} \
                --batch_size {batch size} \
                --is_aug {data augmentation} \
                --dropout_rate {dropout rate} \
                --patience {patience} 
```
## References
[1] Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese
neural networks for one-shot image recognition. In ICML Deep Learning
Workshop, 2015.\
[2] Fares Elmenshawii. Face-detection-dataset. https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset, 2023.