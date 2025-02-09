from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import LFWPairs
from tqdm import tqdm

import argparse
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, image_size=(160, 160)):
        self.data = data

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, label

def prepare_data(transform, batch_size=32, path_save_data="../data/"):
    train_data = LFWPairs(root=path_save_data, split="train", image_set="original", download=True)
    test_data = LFWPairs(root=path_save_data, split="test", image_set="original", download=True)

    train_dataset = CustomDataset(train_data, transform=transform)
    test_dataset = CustomDataset(test_data, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

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

def transform_image(detector, image_size=(160, 160)):
    transform = transforms.Compose([
        FaceDectionTransform(detector),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

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

def main(batch_size, detector_ckpt):
    detector = torch.hub.load("ultralytics/yolov5", "custom", path=detector_ckpt, force_reload=True, verbose=False)
    transform = transform_image(detector)
    train_dataloader, test_dataloader = prepare_data(transform)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--detector_ckpt", type=str, help="Checkpoint of detector", default="../checkpoint/yolov5n/best.pt")

    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        detector_ckpt=args.detector_ckpt
    )