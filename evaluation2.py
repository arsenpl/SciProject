#evaluation of classical algorithms accuracy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score
import numpy as np
import torch
import statistics
import os
import cv2
scale=3 # 1 or 2
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
IMAGE_HEIGHT = 160*scale  # 1280 originally
IMAGE_WIDTH = 240*scale  # 1918 originally
n_clusters=2

def kmeans_segmentation(img):
    # Reshape the image into a vector of pixels
    pixels = np.float32(img.reshape(-1, 3))

    # Apply k-means algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)

    # Assign each pixel to its corresponding cluster
    labels = kmeans.predict(pixels)

    # Reshape the labels back to the original image shape
    labels = np.uint8(labels.reshape(img.shape[:2]))

    # Separate the object and background based on the labels
    object_mask = np.where(labels == 0, 255, 0).astype('uint8')
    background_mask = np.where(labels == 1, 255, 0).astype('uint8')
    other_mask = np.where(labels == 2, 255, 0).astype('uint8')
    img_inverted = cv2.bitwise_not(object_mask)
    background_mask[background_mask == 255] = 1

    return background_mask

def segment(img):
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((2, 2), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    sure_bg = cv2.dilate(closing, kernel, iterations=5)

    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)

    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    return sure_fg

def color_mask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    low = np.array([0, 0, 0])
    high = np.array([215, 51, 51])

    mask = cv2.inRange(img, low, high)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

def check_acc(gt, mask):
    print("gt:",gt)
    print("mask", mask)
    num_correct=0
    num_correct += (gt == mask).sum()
    return num_correct

def dice(pred, true, k = 1):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))
    return dice
def main():
    i=0
    TP=0
    FP=0
    FN=0
    num_correct = 0
    num_pixels = 0
    accTL=[]
    accEL=[]
    dice_scoreL = []
    recallL = []
    precisionL = []

    num_pixelsE = IMAGE_HEIGHT * IMAGE_WIDTH

    for file_name in os.listdir(TEST_IMG_DIR):
        i+=1
        #if i>600:
        file_name=file_name.removesuffix(".jpg")
        img = cv2.imread(TEST_IMG_DIR+file_name+".jpg")
        gt = cv2.imread(TEST_MASK_DIR+file_name+".png", 0)
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, dim, interpolation=cv2.INTER_AREA)

        #Treshold segmentation
        background_mask = segment(img)
        background_mask = background_mask / 255
        background_mask = background_mask.astype(int)

        #K-means segmentation
        #background_mask = kmeans_segmentation(img)

        TP += ((background_mask == 1) & (gt == 1)).sum().item()
        FN += ((background_mask == 0) & (gt == 1)).sum().item()
        FP += ((background_mask == 1) & (gt == 0)).sum().item()

        recall = TP / (TP + FN)
        recallL.append(recall)
        precision = TP / (TP + FP)
        precisionL.append(precision)
        num_correct += np.sum(gt == background_mask)
        num_pixels += IMAGE_HEIGHT*IMAGE_WIDTH


        dice_score = (2 * np.sum(gt * background_mask)) / (np.sum(gt + background_mask) + 1e-8)
        dice_scoreL.append(dice_score)
        accT=num_correct / num_pixels * 100
        accTL.append(accT)


        #showing the results of segmentation with accuracy more than 75%
        if accT<75:
            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            # Plot each image in a separate subplot
            axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[1].imshow(background_mask, cmap="gray")
            axs[2].imshow(gt, cmap="gray")

            # Set the titles for each subplot
            axs[0].set_title('Image')
            axs[1].set_title('Mask')
            axs[2].set_title('Ground truth')

            plt.show()

    print(f"Got total acc {np.mean(accTL):.4f} with {num_correct}/{num_pixels} correct")
    print(f"Dice score: {np.mean(dice_scoreL):.6f}")
    print(f"Recall: {np.mean(recallL):.6f}")
    print(f"Precision: {np.mean(precisionL):.6f}")

    plt.plot(accTL, label='Total accuracy')
    plt.xlabel('Image')
    plt.ylabel('Accuracy %')
    plt.title('Plot of total accuracy')
    plt.legend()
    plt.show()

    plt.plot(dice_scoreL, label='Dice score')
    plt.xlabel('Image')
    plt.ylabel('Dice score')
    plt.title('Plot of dice score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()