import cv2
import numpy as np

# 비디오 캡처 설정 (웹캠)
cap = cv2.VideoCapture(0)

# 빨간색 HSV 범위 설정
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # BGR을 HSV로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 범위에 해당하는 부분을 마스크로 만듦
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # 0-10도 범위의 빨간색
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # 170-180도 범위의 빨간색
    mask = mask1 + mask2

    # 마스크를 적용하여 빨간색 부분만 추출
    red_detected = cv2.bitwise_and(frame, frame, mask=mask)

    # 결과 출력
    cv2.imshow('Original', frame)
    cv2.imshow('Red Detected', red_detected)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 모든 창 닫기
cap.release()
cv2.destroyAllWindows()