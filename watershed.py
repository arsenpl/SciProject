import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the input image
img = cv2.imread('data/train_images/68fcee2be01f_01.jpg')

# Convert the input image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to the correct data type
if gray.dtype != np.uint8:
    gray = gray.astype(np.uint8)

# Apply a threshold to the grayscale image
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Compute the connected components of the thresholded image
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

# Display the result
plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))
plt.show()
