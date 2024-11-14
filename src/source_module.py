from facenet_pytorch import InceptionResnetV1
from PIL import Image

import torchvision.transforms as transforms
import torch.nn as nn

def face_detection(img, detector):
    results = detector(img)
    predictions = results.pandas().xyxy[0]

    if predictions.empty:
        return None

    best_prediction = predictions.loc[predictions["confidence"].idxmax()]
    x_min, y_min, x_max, y_max = int(best_prediction["xmin"]), int(best_prediction["ymin"]), int(best_prediction["xmax"]), int(best_prediction["ymax"])
    return img.crop((x_min, y_min, x_max, y_max))


class FaceDectionTransform:
    def __init__(self, detector):
        self.detector = detector

    def __call__(self, img):
        cropped_img = face_detection(img, self.detector)
        return cropped_img if cropped_img is not None else img

def transform_image(img_path, detector, image_size=(160, 160)):
    img = Image.open(img_path)
    transform = transforms.Compose([
        FaceDectionTransform(detector),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img

class SiameseNeuralNetwork(nn.Module):
    def __init__(self, out_features=64, dropout_rate=0.25):
        super(SiameseNeuralNetwork, self).__init__()
        self.model = InceptionResnetV1(pretrained="vggface2")
        for param in self.model.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

