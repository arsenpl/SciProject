import torch
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
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

# Hyperparameters
#Learning rate for UNET and FCN-8
LEARNING_RATE = 1e-4
#Learning rate for SEGNET
#LEARNING_RATE = 0.005
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 5
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240   # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
#Model architectures: UNET, FCN, SEGNET
MODEL_ARCH="SEGNET"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss

def main():
    print("Training model: "+MODEL_ARCH)
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
        loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif MODEL_ARCH=="FCN":
        model = FCN8s(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif MODEL_ARCH=="SEGNET":
        model = SegNet(in_channels=3, out_channels=1, BN_momentum=0.9).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)

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

    if LOAD_MODEL:
        load_checkpoint(torch.load("models/my_checkpoint"+MODEL_ARCH+".pth.tar"), model)

    print("Introductory evaluation of the model \n")
    check_accuracy(val_loader, model, device=DEVICE)
    print("\n")
    scaler = torch.cuda.amp.GradScaler()
    lossL=[]
    accL=[]
    for epoch in range(NUM_EPOCHS):
        loss=train_fn(train_loader, model, optimizer, loss_fn, scaler)
        lossL.append(loss.item())
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "models/my_checkpoint"+MODEL_ARCH+".pth.tar")

        # check accuracy
        acc=check_accuracy(val_loader, model, device=DEVICE)
        accL.append(acc.item())
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    torch.save(model.state_dict(), "models/model"+MODEL_ARCH+".pt")

    plt.plot(lossL, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Plot of loss in training process of '+ MODEL_ARCH)
    plt.legend()
    plt.show()

    plt.plot(accL, label="Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.title('Plot of accuracy in training process of '+ MODEL_ARCH)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()