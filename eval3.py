#evaluation of classical algorithms accuracy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from sklearn.metrics import f1_score
import numpy as np
import torch
import statistics
import os
import random
import cv2
import csv


TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

test_img_files = os.listdir(TEST_IMG_DIR)
test_mask_files = os.listdir(TEST_MASK_DIR)
train_img_files = os.listdir(TRAIN_IMG_DIR)
train_mask_files = os.listdir(TRAIN_MASK_DIR)
val_img_files = os.listdir(VAL_IMG_DIR)
val_mask_files = os.listdir(VAL_MASK_DIR)

all_img_files = (
    test_img_files + train_img_files + val_img_files
)
all_mask_files = (
    test_mask_files + train_mask_files + val_mask_files
)

#scale=2 # Skala obrazów
n_clusters=2 # Liczba klas w algorytmie k-średnich
n_samples = 50 # Liczba obrazów i masek do wylosowania

#IMAGE_HEIGHT = 160*scale  # 1280 originally
#IMAGE_WIDTH = 240*scale  # 1918 originally

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
    sure_fg=sure_fg/255
    sure_fg=sure_fg.astype(int)
    return sure_fg

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
    #merged_mask = cv2.bitwise_and(mask_r, cv2.bitwise_and(mask_g, mask_b))
    combined_mask = cv2.bitwise_or(mask_r, cv2.bitwise_or(mask_g, mask_b))

    # Inwersja masek
    #inverted_imgG = cv2.bitwise_not(merged_mask)
    out_mask = cv2.bitwise_not(combined_mask)

    out_mask = out_mask / 255
    out_mask = out_mask.astype(int)
    return out_mask #, inverted_imgG


def main():
    # Pobierz listy nazw plików obrazów i masek
    #img_files = os.listdir(IMG_DIR)
    #mask_files = os.listdir(MASK_DIR)
    data = [
        ["Accuracy", "Dice score", "Recall", "Precision"]
    ]
    for i in range(10):
        print("Sprawdzenie #"+str(i))
        selected_group=[]
        # Losowo wybierz n_samples indeksów z zakresu dostępnych plików
        selected_files = random.sample(all_img_files, n_samples)

        for file_name in selected_files:
            #selected_group.append(file_name)
            group_name=file_name[:-6]
            group_files=[]
            #print(group_name)
            for j in range(1,17):
                formatted_name = group_name+"{:02d}".format(j)+".jpg"
                selected_group.append(formatted_name)
            #print(group_files)
        #print(len(all_img_files))
        #print(selected_group)

        for sc in range(1, 3):

            scale = sc  # Skala obrazów
            print("Analiza obrazów w skali: " + str(sc))
            IMAGE_HEIGHT = 160 * scale  # 1280 originally
            IMAGE_WIDTH = 240 * scale  # 1918 originally

            TP = 0
            FP = 0
            FN = 0
            num_correct = 0
            num_pixels = 0
            accTL = []
            accEL = []
            dice_scoreL = []
            recallL = []
            precisionL = []
            num_pixelsE = IMAGE_HEIGHT * IMAGE_WIDTH
            for file_name in selected_group:
                #i+=1
                #if i>600:
                #file_name=file_name.removesuffix(".jpg")
                #img = cv2.imread(TEST_IMG_DIR+file_name+".jpg")
                #gt = cv2.imread(TEST_MASK_DIR+file_name+".png", 0)
                #print(file_name)
                if file_name in test_img_files:
                    img_folder = TEST_IMG_DIR
                    mask_folder = TEST_MASK_DIR
                elif file_name in train_img_files:
                    img_folder = TRAIN_IMG_DIR
                    mask_folder = TRAIN_MASK_DIR
                elif file_name in val_img_files:
                    img_folder = VAL_IMG_DIR
                    mask_folder = VAL_MASK_DIR
                else:
                    continue

                # Wczytaj obraz i maskę
                file_name=file_name.removesuffix(".jpg")
                img = cv2.imread(os.path.join(img_folder, file_name+".jpg"))
                gt = cv2.imread(os.path.join(mask_folder, file_name+".png"), 0)
                dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
                img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                gt = cv2.resize(gt, dim, interpolation=cv2.INTER_AREA)


                #Treshold segmentation/rgbsegmentation
                #background_mask = rgbsegment(img)
                #background_mask = background_mask / 255
                #background_mask = background_mask.astype(int)

                #K-means segmentation
                background_mask = kmeans_segmentation(img)

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
                if accT>99:
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
            data.append([np.mean(accTL),np.mean(dice_scoreL),np.mean(recallL),np.mean(precisionL)])
            '''
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
            '''
        #print(data)
        #print(data[1][1])

    filename = 'KMEANSDATASCALED2.csv'
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


if __name__ == "__main__":
    main()