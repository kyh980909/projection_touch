from transformers import pipeline
import torch
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 경량화 모델
checkpoint = "Intel/dpt-hybrid-midas" #"LiheYoung/depth-anything-small-hf"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

cap = cv2.VideoCapture(0)

# MediaPipe 핸드 스켈레톤 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 테이블 영역의 좌표 (이미지 내 특정 영역을 미리 정의)
TABLE_REGION_X_START = 100
TABLE_REGION_X_END = 220
TABLE_REGION_Y_START = 150
TABLE_REGION_Y_END = 200

# 테이블 영역의 깊이 값 추출을 위한 플래그
table_depth_captured = False
table_depth_value = 0

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (320, 240))

    if not ret:
        break

    # RGB 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 깊이 추정
    depth = depth_estimator(pil_image)
    depth_map = np.array(depth["depth"])
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    # 테이블 영역의 깊이 값을 추출 (최초 한 번만 실행)
    if not table_depth_captured:
        table_depth_area = depth_map[TABLE_REGION_Y_START:TABLE_REGION_Y_END, TABLE_REGION_X_START:TABLE_REGION_X_END]
        table_depth_value = np.mean(table_depth_area)
        table_depth_captured = True

    # MediaPipe를 이용한 핸드 스켈레톤 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 검지 손가락 끝의 좌표 및 깊이 정보 추출
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # 검지 손가락 끝의 좌표 (Landmark ID 8)
            index_finger_tip = hand_landmarks.landmark[8]
            finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # 손가락 끝이 테이블 영역 내에 있는지 확인
            if (TABLE_REGION_X_START <= finger_x <= TABLE_REGION_X_END and
                TABLE_REGION_Y_START <= finger_y <= TABLE_REGION_Y_END):
                
                # 검지 손가락 끝의 깊이 추출
                finger_depth = depth_map[finger_y, finger_x]

                # 터치 판정: 테이블 영역의 깊이와 유사한지 확인 (예: 오차 범위 10)
                if abs(finger_depth - table_depth_value) < 10:
                    cv2.putText(frame, 'Touch Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손가락 끝에 원 그리기
            cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1)

    # 테이블 영역에 사각형 그리기
    cv2.rectangle(frame, (TABLE_REGION_X_START, TABLE_REGION_Y_START), 
                         (TABLE_REGION_X_END, TABLE_REGION_Y_END), (255, 0, 0), 2)

    # 결과 출력
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Depth Map', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()