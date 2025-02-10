from SNNs import SiameseNeuralNetwork
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

# import pathlib
# pathlib.PosixPath = pathlib.WindowsPath

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

def prepare_data(transform, path_save_data, batch_size=32):
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

def eval_model(test_dataloader, model, criterion, optimizer):
    model.eval()

    total_loss_val = 0
    with torch.no_grad():
        for (image1, image2, label) in test_dataloader:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)
            output = model(image1, image2)
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

        for (image1, image2, label) in progress_bar:
            image1, image2, label = image1.to(device), image2.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image1, image2)
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

def main(num_epochs, lr, batch_size, patience, model_ckpt, detector_ckpt, path_save_data):
    detector = torch.hub.load("ultralytics/yolov5", "custom", path=detector_ckpt, force_reload=True, verbose=False)
    transform = transform_image(detector)
    train_dataloader, test_dataloader = prepare_data(transform, path_save_data, batch_size)

    model = SiameseNeuralNetwork()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    train_model(num_epochs, train_dataloader, test_dataloader, model, criterion, optimizer, device, model_ckpt, patience)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyper-parameters for training")

    parser.add_argument("--num_epochs", type=int, help="Num epochs for training", default=100)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.1)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--patience", type=int, help="The number of epochs without improvement before stopping training", default=10)
    parser.add_argument("--model_ckpt", type=str, help="Checkpoint folder of model save last, best model and ...", default="../checkpoint/model/")
    parser.add_argument("--detector_ckpt", type=str, help="Checkpoint of detector", default="../checkpoint/yolov5n/best.pt")
    parser.add_argument("--path_save_data", type=str, help="Directory save data", default="../data/")

    args = parser.parse_args()

    main(
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        model_ckpt=args.model_ckpt,
        detector_ckpt=args.detector_ckpt,
        path_save_data=args.path_save_data
    )
