import numpy as np
import cv2
from ultralytics import YOLO

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    return clahe_bgr

model = YOLO('last.pt')

original_image = cv2.imread('image2.jpg')
original_image=apply_clahe(original_image)
resized_image = cv2.resize(original_image, (640, 640))
results = model.predict(resized_image,show=True)
image_height = 640
image_width = 640

binary_mask = np.zeros((image_height, image_width), dtype=np.uint8)

for r in results:
    if r.masks is not None:
        for i, mask_points in enumerate(r.masks.xy):
            mask_points = np.array(mask_points, dtype=np.int32).reshape((-1, 1, 2))

            cv2.fillPoly(binary_mask, [mask_points], 255)
        
            color = (0, 255, 0) 
            cv2.polylines(original_image, [mask_points], isClosed=True, color=color, thickness=2)

cv2.imshow("Binary Mask", binary_mask)
cv2.imshow("Original Image with Masks", resized_image) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('mask.jpg',binary_mask)