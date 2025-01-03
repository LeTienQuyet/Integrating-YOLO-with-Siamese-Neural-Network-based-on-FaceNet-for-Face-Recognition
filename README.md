# Integrating-YOLO-with-Siamese-Neural-Network-based-on-FaceNet-for-Face-Recognition
This repo utilizes **Siamese Neural Networks (SNNs)** combined with **YOLO** to enhance the effectiveness of face recognition.
![pipeline](./image/pipeline.png)
## Detector
* Before inputting the image into the FaceNet model, we deploy a detector to identify the location most likely to contain a face, specifically the YOLOv5n model. This process allows us to accurately determine the area where a face may appear, thereby enhancing the accuracy and reliability of the model.

* After identifying the face location, we will use this region as input for the FaceNet model instead of the original image. The YOLOv5n model has been further trained on the [**Face-Detection-Dataset**](https://www.kaggle.com/datasets/fareselmenshawii/face-detection-dataset) for a total of 30 epochs to significantly improve its face detection capabilities.
## Reference
Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Siamese
neural networks for one-shot image recognition. In ICML Deep Learning
Workshop, 2015.