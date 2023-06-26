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
scale=3# 1 or 2
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
IMAGE_HEIGHT = 160*scale  # 1280 originally
IMAGE_WIDTH = 240*scale  # 1918 originally
n_clusters=2

def kmeans_segmentation(img):
    # Przekształcenie obrazu w wektor pikseli
    pixels = np.float32(img.reshape(-1, 3))

    # Zastosowanie algorytmu k-średnich
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)

    # Przydzielanie każdego piksela do odpowiedniej grupy
    labels = kmeans.predict(pixels)

    # Przekształcenie wektora etykiet do rozmiarów obrazu
    labels = np.uint8(labels.reshape(img.shape[:2]))

    # Oddzielenie obrazu i tła bazując na etykiecie
    object_mask = np.where(labels == 0, 255, 0).astype('uint8')
    background_mask = np.where(labels == 1, 255, 0).astype('uint8')
    other_mask = np.where(labels == 2, 255, 0).astype('uint8')
    img_inverted = cv2.bitwise_not(object_mask)
    background_mask[background_mask == 255] = 1

    return background_mask

def segment(img):
    # Przekształcenie obrazu w format BGR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Wykonania progowania z automatycznym wyznaczeniem optymalnego progu
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Tworzenie jądra i wykonania zamknięcia morfologicznego
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    sure_bg = cv2.dilate(closing, kernel, iterations=5)

    # Obliczanie transformacji odległościowej i ponowne progowanie
    dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)

    return sure_fg

def showimages(r, g, b, gray):
    fig, axs = plt.subplots(1, 4, figsize=(10, 5))
    # Plot each image in a separate subplot
    axs[0].imshow(r, cmap="gray")
    axs[1].imshow(g, cmap="gray")
    axs[2].imshow(b, cmap="gray")
    axs[3].imshow(gray, cmap="gray")

    # Set the titles for each subplot
    axs[0].set_title('red')
    axs[1].set_title('green')
    axs[2].set_title('blue')
    axs[3].set_title('gray')

    plt.show()
def rgbsegment(img):
    # Podział obrazu na kanały kolorowe
    b, g, r = cv2.split(img)

    # Tworzenie masek dla każdego kanału
    mask_r = np.zeros_like(r)
    mask_g = np.zeros_like(g)
    mask_b = np.zeros_like(b)

    # Progowanie dla każdego kanału
    ret, thresh_r = cv2.threshold(r, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh_g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ret, thresh_b = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_r[r > thresh_r] = 255
    mask_g[g > thresh_g] = 255
    mask_b[b > thresh_b] = 255

    # Połaczenie kanałów kolorowych
    merged_mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))
    combined_mask = cv2.bitwise_or(mask_r, cv2.bitwise_or(mask_g, mask_b))

    # Inwersja masek
    inverted_imgG = cv2.bitwise_not(merged_mask)
    inverted_img = cv2.bitwise_not(combined_mask)


    return inverted_img, inverted_imgG

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

    TP1 = 0
    FP1 = 0
    FN1 = 0
    num_correct1 = 0
    num_pixels1 = 0
    accTL1 = []
    accEL1 = []
    dice_scoreL1 = []
    recallL1 = []
    precisionL1 = []

    num_pixelsE = IMAGE_HEIGHT * IMAGE_WIDTH

    for file_name in os.listdir(TEST_IMG_DIR):
        i+=1
        if i>0:
            file_name=file_name.removesuffix(".jpg")
            img = cv2.imread(TEST_IMG_DIR+file_name+".jpg")
            gt = cv2.imread(TEST_MASK_DIR+file_name+".png", 0)
            dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, dim, interpolation=cv2.INTER_AREA)

            #Treshold segmentation
            background_mask, background_maskG = rgbsegment(img)
            background_mask = background_mask / 255
            background_mask = background_mask.astype(int)
            background_maskG = segment(img)
            background_maskG = background_maskG / 255
            background_maskG = background_maskG.astype(int)

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

            TP1 += ((background_maskG == 1) & (gt == 1)).sum().item()
            FN1 += ((background_maskG == 0) & (gt == 1)).sum().item()
            FP1 += ((background_maskG == 1) & (gt == 0)).sum().item()

            recall1 = TP1 / (TP1 + FN1)
            recallL1.append(recall1)
            precision1 = TP1 / (TP1 + FP1)
            precisionL1.append(precision1)
            num_correct1 += np.sum(gt == background_maskG)
            num_pixels1 += IMAGE_HEIGHT * IMAGE_WIDTH

            dice_score1 = (2 * np.sum(gt * background_maskG)) / (np.sum(gt + background_maskG) + 1e-8)
            dice_scoreL1.append(dice_score1)
            accT1 = num_correct1 / num_pixels1 * 100
            accTL1.append(accT1)


            #showing the results of segmentation with accuracy more than 75%
            if accT>95:
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

    print(f"Got total acc {np.mean(accTL1):.4f} with {num_correct1}/{num_pixels1} correct")
    print(f"Dice score: {np.mean(dice_scoreL1):.6f}")
    print(f"Recall: {np.mean(recallL1):.6f}")
    print(f"Precision: {np.mean(precisionL1):.6f}")

    plt.plot(accTL, color='b', label='Total accuracy OR')
    plt.plot(accTL1, color='r', label='Total accuracy AND')
    plt.xlabel('Image')
    plt.ylabel('Accuracy %')
    plt.title('Plot of total accuracy')
    plt.legend()
    plt.show()

    plt.plot(dice_scoreL, color='b', label='Dice score')
    plt.plot(dice_scoreL1, color='r',label='Dice score')
    plt.xlabel('Image')
    plt.ylabel('Dice score')
    plt.title('Plot of dice score')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()