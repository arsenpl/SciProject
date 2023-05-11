#unilities
import numpy as np
import torch
import torchvision
from sklearn.metrics import f1_score
from dataset import CarvanaDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="models/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    test_dir,
    test_maskdir,
    batch_size,
    train_transform,
    val_transform,
    test_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = CarvanaDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = CarvanaDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = CarvanaDataset(
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, val_loader, test_loader

def check_accuracy(loader, loss_fn,  model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    f1_scoreS = 0
    model.eval()
    lossL=[]
    i=0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            i+=1

            preds = torch.sigmoid(model(x))
            loss=loss_fn(preds,y)
            lossL.append(loss.item())
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    acc=num_correct/num_pixels*100
    print(
        f"Got {num_correct}/{num_pixels} with acc {acc:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    print(i)
    model.train()
    return acc, lossL


def check_accuracy2(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_scoreL = []
    dice_scoreS=0
    accL = []
    model.eval()
    recallL=[]
    precisionL=[]
    precisionS=0
    TP=0
    FP=0
    FN=0


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            # print(x.shape," ",y.shape)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score = (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )
            dice_scoreS+=dice_score
            dice_scoreL.append(dice_score.item())
            acc = num_correct / num_pixels * 100
            #print(acc, dice_score)
            accL.append(acc.item())

            TP += ((preds == 1) & (y == 1)).sum().item()
            FN += ((preds == 0) & (y == 1)).sum().item()
            FP += ((preds == 1) & (y == 0)).sum().item()

            recall = TP / (TP + FN)
            recallL.append(recall)
            precision = TP / (TP+FP)
            precisionL.append(precision)
            precisionS+=precision
            #print(recall)

    mean_recall=np.mean(recallL)*100
    print(f"Recall: {mean_recall:.10f}")
    mean_precision=np.mean(precisionL)*100
    print(f"Precision: {(precisionS/len(loader))*100:.10f}")
    mean_accuracy=np.mean(accL)
    print(f"Got {num_correct}/{num_pixels} with acc {mean_accuracy:.4f}")
    mean_dice_score=(dice_scoreS/len(loader))*100
    print(f"Dice score: {mean_dice_score}")
    # print(f"F1 score: {f1_scoreS / len(loader)}")
    #model.train()
    return mean_accuracy, mean_dice_score.item(), mean_recall, mean_precision, accL, dice_scoreL

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        #torchvision.utils.save_image(x, f"{folder}{idx}.png")

    model.train()