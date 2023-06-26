#training process
import torch
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import time
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

LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 5
NUM_WORKERS = 2
scale=1 #1 or 2
IMAGE_HEIGHT = 160*scale  # 1280 originally
IMAGE_WIDTH = 240*scale   # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
#Model architectures: UNET, FCN, SEGNET
MODEL_ARCH="SEGNETproba"

def train_fn(train_loader, val_loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(train_loader)
    num_correct = 0
    num_pixels = 0
    lossT=[]
    accT=[]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            lossT.append(loss.item())
            predictions = torch.sigmoid(predictions)
            num_correct = (predictions.round() == targets).sum()

            num_pixels = torch.numel(predictions)

            acc = num_correct / num_pixels * 100
            accT.append(acc.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=acc.item())
    loop = tqdm(val_loader)
    model.eval()
    num_correct = 0
    num_pixels = 0
    lossV = []
    accV = []
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            lossV.append(loss.item())
            predictions = torch.sigmoid(predictions)
            num_correct = (predictions.round() == targets).sum()

            num_pixels = torch.numel(predictions)

            acc = num_correct / num_pixels * 100
            accV.append(acc.item())

            # update tqdm loop
        loop.set_postfix(loss=loss.item(), accuracy=acc.item())
    return lossT, accT, lossV, accV

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

    if MODEL_ARCH.__contains__("UNET"):
        model = UNET(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif MODEL_ARCH.__contains__("FCN"):
        model = FCN8s(in_channels=3, out_channels=1).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        #loss_fn = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif MODEL_ARCH.__contains__("SEGNET"):
        model = SegNet(in_channels=3, out_channels=1, BN_momentum=0.9).to(DEVICE)
        loss_fn = nn.BCEWithLogitsLoss()
        #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
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
    check_accuracy(val_loader,loss_fn, model, device=DEVICE)
    print("\n")
    scaler = torch.cuda.amp.GradScaler()
    lossT=[]
    lossV=[]
    accT=[]
    accV=[]
    starttime=time.time()
    for epoch in range(NUM_EPOCHS):
        print("Epoch: ", epoch+1)
        loss_train,acc_train, loss_val, acc_val=train_fn(train_loader,val_loader, model, optimizer, loss_fn, scaler)
        lossT+=loss_train
        accT+=acc_train
        lossV+=loss_val
        accV+=acc_val
        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, "models/my_checkpoint"+MODEL_ARCH+".pth.tar")

        # check accuracy
        #acc, loss_val=check_accuracy(val_loader, loss_fn, model, device=DEVICE)
        #lossV+=loss_val
        #accL.append(acc.item())
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    torch.save(model.state_dict(), "models/model"+MODEL_ARCH+".pt")
    endtime=time.time()
    traintime=endtime-starttime
    with open(MODEL_ARCH+"loss.txt", "w") as f:
        for s in lossT:
            f.write(str(s) + "\n")

    print("Time of training: "+str(traintime/60)+" minute")
    plt.plot(lossT, label='Loss')
    #plt.plot(lossV, label='Validation')
    plt.xlabel('Batches')
    plt.ylabel('Loss')
    plt.title('Plot of loss in training process of '+ MODEL_ARCH[:-1])
    plt.legend()
    plt.show()

    with open("measurements/"+MODEL_ARCH+"acc.txt", "w") as f:
        for s in accT:
            f.write(str(s) + "\n")
    plt.plot(accT, label="Accuracy")
    #plt.plot(accV, label="Validation")
    plt.xlabel('Batches')
    plt.ylabel('Accuracy %')
    plt.title('Plot of accuracy in training process of '+ MODEL_ARCH[:-1])
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()