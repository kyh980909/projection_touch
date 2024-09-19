from transformers import pipeline
import torch
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 경량화 모델
checkpoint = "LiheYoung/depth-anything-small-hf"
depth_estimator = pipeline("depth-estimation", model=checkpoint)

cap = cv2.VideoCapture(0)

# MediaPipe 핸드 스켈레톤 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

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

    # MediaPipe를 이용한 핸드 스켈레톤 추출
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # 손가락 끝의 위치 및 깊이 정보 추출
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 인덱스 핑거(손가락 끝)의 좌표 추출 (Landmark ID 8번이 인덱스 핑거)
            index_finger_tip = hand_landmarks.landmark[8]
            h, w, _ = frame.shape
            finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # 손가락 끝의 깊이 값 추출
            finger_depth = depth_map[finger_y, finger_x]

            # 터치 판정 기준 (예: 특정 깊이 이하일 때 터치로 간주)
            if finger_depth < 50:  # 이 값은 환경에 맞게 조정
                cv2.putText(frame, 'Touch Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손가락 끝에 원 그리기
            cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), -1)

    # 결과 출력
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Depth Map', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()