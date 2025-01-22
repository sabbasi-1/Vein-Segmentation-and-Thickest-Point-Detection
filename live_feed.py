import numpy as np
import cv2

CAMERA_INDEX = 0

def process_frame(original_image):
    processed_image = original_image
    return processed_image


cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame!")
        break

    cv2.imshow("Original Stream", frame)
    processed = process_frame(frame)
    cv2.imshow("Processed", processed)

    key_press = cv2.waitKey(1)
    if key_press == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
