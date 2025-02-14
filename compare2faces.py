from src.SNNs import SiameseNeuralNetwork
from src.train import transform_img
from PIL import Image

import argparse
import torch
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import pathlib
pathlib.PosixPath = pathlib.WindowsPath

def extract_feature(model, img_path, transform):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    feature = model.base_model(img_tensor)
    return feature

def main(img1_path, img2_path, noti, model_ckpt, detector_ckpt, force_reload):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    detector = torch.hub.load("ultralytics/yolov5", "custom", path=detector_ckpt, force_reload=force_reload, verbose=False)

    model = SiameseNeuralNetwork(dropout_rate=0.5)
    model.load_state_dict(torch.load(model_ckpt, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transform_img(detector)

    start_time = time.time()

    feature_1 = extract_feature(model, img1_path, transform)
    feature_2 = extract_feature(model, img2_path, transform)

    distance_feature = torch.abs(feature_1 - feature_2)
    output = model.fc(distance_feature)
    result = torch.sigmoid(output).item()

    end_time = time.time()

    if noti:
        if result >= 0.62:
            print("Same person !!!")
        else:
            print("Diff person !!!")
    else:
        print(f"Probability of the same person = {result:.4f}.")

    print(f"Predicted time: {end_time - start_time:.4f}s")
    return result

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
    parser = argparse.ArgumentParser(description="Hyper-parameters for prediction")

    parser.add_argument("--img1_path", type=str, help="Image 1st filename in folder `image`", default="./image/suzy_1.jpg")
    parser.add_argument("--img2_path", type=str, help="Image 2nd filename in folder `image`", default="./image/suzy_2.jpeg")
    parser.add_argument("--noti", type=str2bool, nargs="?", const=True, help="Notification same/diff person between", default=True)
    parser.add_argument("--model_ckpt", type=str, help="Checkpoint of model", default="./checkpoint/model/best_model.pt")
    parser.add_argument("--detector_ckpt", type=str, help="Checkpoint of detector", default="./checkpoint/yolov5n/best.pt")
    parser.add_argument("--force_reload", type=str2bool, nargs="?", const=False, help="Force reload YOLO", default=False)

    args = parser.parse_args()

    result = main(
        img1_path=args.img1_path,
        img2_path=args.img2_path,
        noti=args.noti,
        model_ckpt=args.model_ckpt,
        detector_ckpt=args.detector_ckpt,
        force_reload=args.force_reload
    )