from ultralytics import YOLO
import cv2
img_path = 'image2.jpg' 
image = cv2.imread(img_path)

resized_image = cv2.resize(image, (640, 640))
def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    clahe_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)

    return clahe_bgr

image=apply_clahe(resized_image)
model=YOLO('last.pt')

results=model(image,show=True,save=True)
for r in results:
    if r.masks is not None:
        print(f"Number of masks detected: {len(r.masks.xy)}")