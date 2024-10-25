import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import *

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'width: {width}, height: {height}')

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out2 = cv2.VideoWriter('test.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        print("Failed to capture frame from camera. Exiting.")
        break

    # corners, result_image = find_green_corners(img)

    # 결과 출력
    # print("Corners of the green area:", corners)
    cv2.imshow('img', img)

    out2.write(img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out2.release()
cv2.destroyAllWindows()