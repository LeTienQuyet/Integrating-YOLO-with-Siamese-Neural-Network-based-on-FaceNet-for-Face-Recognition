from src.SNNs import SiameseNeuralNetwork
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import LFWPairs
from tqdm import tqdm

import os
import argparse
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random

# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath

def random_brightness(img, factor_range=(0.5, 1.5)):
    factor = random.uniform(*factor_range)
    return transforms.functional.adjust_brightness(img, factor)

def random_horizontal_flip(img):
    if random.random() > 0.5:
        img = transforms.functional.hflip(img)
    return img

class CustomTransform:
    def __init__(self, brightness_range=(0.5, 1.5)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        img = random_brightness(img, self.brightness_range)
        img = random_horizontal_flip(img)
        return img

class CustomDataset(Dataset):
    def __init__(self, data, transform=None, img_size=(160, 160)):
        self.data = data

        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, label

def prepare_data(transform_train, transform_test, path_data, batch_size=32):
    train_data = LFWPairs(root=path_data, split="train", image_set="original", download=True)
    test_data = LFWPairs(root=path_data, split="test", image_set="original", download=True)

    train_dataset = CustomDataset(train_data, transform=transform_train)
    test_dataset = CustomDataset(test_data, transform=transform_test)

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

def transform_img(detector, img_size=(160, 160)):
    transform = transforms.Compose([
        FaceDectionTransform(detector),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def transform_img_with_augmentation(detector, img_size=(160, 160)):
    transform = transforms.Compose([
        FaceDectionTransform(detector),
        CustomTransform(brightness_range=(0.8, 1.2)),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def eval_model(test_dataloader, model, criterion, optimizer):
    model.eval()

    total_loss_val = 0
    with torch.no_grad():
        for (img1, img2, label) in test_dataloader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output = model(img1, img2)
            loss = criterion(output, label)
            total_loss_val += loss.item()

    loss_val = total_loss_val / len(test_dataloader.dataset)
    return loss_val

def train_model(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device, model_ckpt, patience):
    if not os.path.exists(model_ckpt):
        os.mkdir(model_ckpt)
        print(f"Folder {model_ckpt} has been created!")
    else:
        print(f"Folder {model_ckpt} already exists!")

    min_loss = float("inf")
    train_losses = []
    val_losses = []

    early_stop_counter = 0
    best_epoch = 0

    model.to(device)
    for epoch in range(num_epochs):
        model.train()

        total_loss_train = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", unit="batch", colour="RED")

        for (img1, img2, label) in progress_bar:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(img1, img2)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            total_loss_train += loss.item()

        loss_train = total_loss_train / len(train_dataloader.dataset)
        train_losses.append(loss_train)

        loss_val = eval_model(test_dataloader, model, criterion, optimizer)
        val_losses.append(loss_val)

        print(f"Epoch {epoch+1}/{num_epochs}: Train loss = {loss_train:.4f} - Val loss = {loss_val:.4f}")

        if loss_val < min_loss:
            min_loss = loss_val
            torch.save(model.state_dict(), os.path.join(model_ckpt, "best_model.pt"))
            print(f"Best model saved at Epoch {epoch+1}\n")
            early_stop_counter = 0
            best_epoch = epoch + 1
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs.")

        if early_stop_counter >= patience:
            print(f"Early stopping at Epoch {epoch+1}.")
            print(f"Best Val loss = {min_loss:.4f} at Epoch {best_epoch}.")
            break

        torch.save(model.state_dict(), os.path.join(model_ckpt, "last_model.pt"))

    epochs = range(1, num_epochs)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(model_ckpt, "train_eval_loss.png"), dpi=300, bbox_inches='tight')

def main(num_epochs, lr, batch_size, is_aug, dropout_rate, patience, model_ckpt, detector_ckpt, path_data):
    detector = torch.hub.load("ultralytics/yolov5", "custom", path=detector_ckpt, force_reload=True, verbose=False)
    transform_test = transform_img(detector)
    transform_train = transform_img_with_augmentation(detector) if is_aug else transform_test

    train_dataloader, test_dataloader = prepare_data(transform_train, transform_test, path_data, batch_size)

    model = SiameseNeuralNetwork(dropout_rate=dropout_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    train_model(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device, model_ckpt, patience)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--num_epochs", type=int, help="Num epochs for training", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--is_aug", type=str2bool, nargs="?", const=False, help="Is data augmentation used for training?", default=False)
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate", default=0.5)
    parser.add_argument("--patience", type=int, help="The number of epochs without improvement before stopping training", default=10)
    parser.add_argument("--model_ckpt", type=str, help="Checkpoint folder of model save last, best model and ...", default="../checkpoint/model/")
    parser.add_argument("--detector_ckpt", type=str, help="Checkpoint of detector", default="../checkpoint/yolov5n/best.pt")
    parser.add_argument("--path_data", type=str, help="Directory download/upload data for training", default="../data/")

    args = parser.parse_args()

    main(
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        is_aug=args.is_aug,
        dropout_rate=args.dropout_rate,
        patience=args.patience,
        model_ckpt=args.model_ckpt,
        detector_ckpt=args.detector_ckpt,
        path_data=args.path_data
    )
