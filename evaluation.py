#evaluation of neural network segmentation models accuracy
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modelUNET import UNET
from modelFCN import FCN8s
from modelSegNet import SegNet
import matplotlib.pyplot as plt
import csv
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    check_accuracy2,
    save_predictions_as_imgs,
)

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
scale=1 #1 or 2
IMAGE_HEIGHT = 160*scale  # 1280 originally
IMAGE_WIDTH = 240*scale   # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
#Model architectures: UNET, FCN, SEGNET
MODEL_ARCHS=["UNET"]

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    test_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        test_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    for MODEL_ARCH in MODEL_ARCHS:
        print(MODEL_ARCH)
        if MODEL_ARCH.__contains__("UNET"):
            model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        elif MODEL_ARCH.__contains__("FCN"):
            model = FCN8s(in_channels=3, out_channels=1).to(DEVICE)
        elif MODEL_ARCH.__contains__("SEGNET"):
            model = SegNet(in_channels=3, out_channels=1, BN_momentum=0.9).to(DEVICE)

        model.load_state_dict(torch.load("models/model"+MODEL_ARCH+".pt"))
        model.eval()
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        #filename = 'measurements/' + MODEL_ARCH + 'data.csv'
        filename = MODEL_ARCH+'2data.csv'
        if LOAD_MODEL:
            load_checkpoint(torch.load("models/my_checkpoint"+MODEL_ARCH+".pth.tar"), model)
        data=[
            ["Accuracy", "Dice score", "Recall", "Precision"]
        ]
        for i in range(1):
            print(i)
            accuracy, dice_score, recall, precision, accL, dice_scoreL=check_accuracy2(test_loader, model, device=DEVICE)
            data.append([accuracy, dice_score, recall,precision])
        print(data)

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            for row in data:
                writer.writerow(row)
        """
        plt.plot(dice_scoreL, label='Accuracy')
        plt.xlabel('Images')
        plt.ylabel('Acccuracy %')
        plt.title('Plot of accuracy for every image with model ' + MODEL_ARCH)
        plt.legend()
        plt.show()
        """

if __name__ == "__main__":
    main()