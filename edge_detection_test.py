import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'width: {width}, height: {height}')

idx = 1

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    # corners, result_image, hsv, mask = find_green_corners(img)
    corners, result_image = find_green_corners(img)
    
    if corners:
        print("Corners of the green area:", corners)
        cv2.imwrite(f'./module_camera_edge_result/edge_detection_result{idx}.jpg', result_image)
        idx += 1

    # 결과 출력
    cv2.imshow('img', result_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()