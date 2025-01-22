import os
import random
import cv2
from tqdm import tqdm

# Function to apply Gaussian blur to an image
def apply_blur(image_path, output_path):
    img = cv2.imread(image_path)
    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply slight Gaussian blur
    cv2.imwrite(output_path, blurred_img)

img='000000000139.jpg'
out='image.jpg'
apply_blur(img,out)