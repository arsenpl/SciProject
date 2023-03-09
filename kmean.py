from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

# Load the image
img = cv2.imread('data/train_images/68fcee2be01f_01.jpg')
gt = cv2.imread('data/train_masks/68fcee2be01f_01.png')
# Reshape the image into a vector of pixels
pixels = np.float32(img.reshape(-1, 3))

# Define the number of clusters
k = 2

# Apply k-means algorithm
kmeans = KMeans(n_clusters=k, random_state=0).fit(pixels)

# Assign each pixel to its corresponding cluster
labels = kmeans.predict(pixels)

# Reshape the labels back to the original image shape
labels = np.uint8(labels.reshape(img.shape[:2]))

# Separate the object and background based on the labels
object_mask = np.where(labels == 0, 255, 0).astype('uint8')
background_mask = np.where(labels == 1, 255, 0).astype('uint8')
other_mask = np.where(labels == 2, 255, 0).astype('uint8')
img_inverted = cv2.bitwise_not(object_mask)

# Show the segmented image and masks
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
# Plot each image in a separate subplot
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[1].imshow(cv2.cvtColor(background_mask, cv2.COLOR_BGR2RGB))
axs[2].imshow(cv2.cvtColor(gt, cv2.COLOR_BGR2RGB))

# Set the titles for each subplot
axs[0].set_title('Image')
axs[1].set_title('Mask')
axs[2].set_title('Ground truth')


plt.show()
