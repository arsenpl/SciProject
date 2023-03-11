from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import numpy as np
import torch
import statistics
import os
import cv2

TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240   # 1918 originally
n_clusters=2

def segment(img):
    b, g, r = cv2.split(img)
    rgb_img = cv2.merge([r, g, b])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((2, 2), np.uint8)
    # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel, iterations=5)
    return sure_bg

def watershed_segmentation(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]


    return markers

def kmeans_segmentation2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img)
    twoDimage = img.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    attempts = 10
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    return result_image

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

def tresh_seg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
    edges = cv2.dilate(cv2.Canny(thresh, 0, 255), None)
    cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
    mask = np.zeros((256, 256), np.uint8)
    masked = cv2.drawContours(mask, [cnt], -1, 255, -1)
    return masked

def otsu(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    thresh = threshold_otsu(img_gray)
    img_otsu = img_gray < thresh

    def filter_image(image, mask):
        r = image[:, :, 0] * mask
        g = image[:, :, 1] * mask
        b = image[:, :, 2] * mask

        return np.dstack([r, g, b])

    filtered = filter_image(img, img_otsu)
    return filtered

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

def main():
    i=0
    accTL=[]
    accEL=[]
    num_pixelsE = IMAGE_HEIGHT * IMAGE_WIDTH
    num_correct = 0
    num_pixels = 0

    for file_name in os.listdir(TEST_IMG_DIR):
        i+=1
        #if i>25:
        #    break
        file_name=file_name.removesuffix(".jpg")
        img = cv2.imread(TEST_IMG_DIR+file_name+".jpg")
        gt = cv2.imread(TEST_MASK_DIR+file_name+".png", 0)
        dim = (IMAGE_WIDTH, IMAGE_HEIGHT)
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, dim, interpolation=cv2.INTER_AREA)


        background_mask = kmeans_segmentation(img)

        num_correct += np.sum(gt == background_mask)
        num_pixels += IMAGE_HEIGHT*IMAGE_WIDTH
        num_correctE = np.sum(gt == background_mask)

        """
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
        """
        accT=num_correct / num_pixels * 100
        accTL.append(accT)
        print("Image: ", i)
        print(
            f"Got total acc {accT:.2f} with {num_correct}/{num_pixels} correct"
        )
        accE=num_correctE / num_pixelsE * 100
        accEL.append(accE)
        print(
            f"Got acc {accE:.2f} for image with {num_correctE}/{num_pixelsE} correct"
        )
    meanAcc=statistics.mean(accEL)
    print(meanAcc)
    plt.plot(accTL, label='Total accuracy')
    plt.plot(accEL, label="Accuracy for each image")

    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Plot of accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()