import cv2
import numpy as np

# 이미지 불러오기
cap = cv2.VideoCapture(0)  # 웹캠으로부터 영상을 가져옴

while True:
    ret, frame = cap.read()  # 웹캠 프레임 읽기
    if not ret:
        break

    # BGR 이미지를 HSV 이미지로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[: ,:, 0]=0
    hsv[: ,:, 2]=0
    
    #s channel 
    cv2.imshow("s channel", hsv)
    cv2.imshow("frame", frame)
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()