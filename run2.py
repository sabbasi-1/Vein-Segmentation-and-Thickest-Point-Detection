import cv2
import numpy as np
import matplotlib.pyplot as plt

mask = cv2.imread('mask.jpg', cv2.IMREAD_GRAYSCALE)
original_image = cv2.imread('image2.jpg')

resized_mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_LINEAR)

original_resized = cv2.resize(original_image, (1280, 720), interpolation=cv2.INTER_LINEAR)
_, binary_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)

skeleton = cv2.ximgproc.thinning(binary_mask)

dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)

output_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
cv2.circle(output_mask, max_loc, 5, (255, 0, 0), -1)  
cv2.circle(original_resized, max_loc, 5, (255, 0, 0), -1) 

plt.figure(figsize=(12, 7))
plt.subplot(1, 2, 1)
plt.imshow(output_mask)
plt.title("Resized Mask with Thickest Point")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(original_resized, cv2.COLOR_BGR2RGB))
plt.title("Original Image with Thickest Point")
plt.axis('off')

plt.show()
cv2.imwrite("thickest_point_on_mask.png", output_mask)
cv2.imwrite("thickest_point_on_original.png", original_resized)

print(f"Thickest point location: {max_loc}, with thickness of {max_val} pixels")