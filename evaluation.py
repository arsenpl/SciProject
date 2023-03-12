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
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
#LEARNING_RATE = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240   # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
#Model architectures: UNET, FCN, SEGNET
MODEL_ARCH="SEGNET"

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
    if MODEL_ARCH=="UNET":
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL_ARCH=="FCN":
        model = FCN8s(in_channels=3, out_channels=1).to(DEVICE)
    elif MODEL_ARCH=="SEGNET":
        model = SegNet(in_channels=3, out_channels=1, BN_momentum=0.9).to(DEVICE)

    model.load_state_dict(torch.load("models/model"+MODEL_ARCH+".pt"))
    model.eval()
    #model = FCN8s(in_channels=3, out_channels=1).to(DEVICE)
    #model = SegNet(in_channels=3, out_channels=1, BN_momentum=0.9).to(DEVICE)

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
    """
    img, label = next(iter(val_loader))
    image = img[0].squeeze()
    print(image)
    plt.imshow(image.permute(1,2,0), cmap="gray")
    plt.show()
    """
    if LOAD_MODEL:
        load_checkpoint(torch.load("models/my_checkpoint"+MODEL_ARCH+".pth.tar"), model)
    load_checkpoint(torch.load("models/my_checkpoint"+MODEL_ARCH+".pth.tar"), model)
    check_accuracy(val_loader, model, device=DEVICE)


if __name__ == "__main__":
    main()