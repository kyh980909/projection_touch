from transformers import pipeline
import torch
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 경량화 모델
checkpoint = "LiheYoung/depth-anything-small-hf"
depth_estimator = pipeline("depth-estimation", model=checkpoint, device=device)

cap = cv2.VideoCapture(0)

# MediaPipe 핸드 스켈레톤 모델 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# 테이블 영역의 네 모서리 좌표 (기본값)
table_points = np.float32([
    [400, 150],  # 좌측 상단
    [420, 150],  # 우측 상단
    [490, 200],   # 좌측 하단
    [430, 200]   # 우측 하단
])

# 변환 후 출력 이미지 크기
transformed_size = (320, 240)
transformed_points = np.float32([
    [0, 0],                   
    [transformed_size[0], 0],  
    [0, transformed_size[1]],  
    [transformed_size[0], transformed_size[1]] 
])

# 테이블 영역의 깊이 값
table_depth_value = 0

# 포인트를 조정 중인지 확인
dragging_point = None

# 마우스 이벤트 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global dragging_point, table_points, table_depth_value, transformed_depth_map

    # 좌표 근처에서 클릭했을 때 해당 포인트를 선택
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, point in enumerate(table_points):
            if np.linalg.norm(np.array([x, y]) - point) < 10:
                dragging_point = i
                break

    # 마우스가 움직이는 동안 선택된 포인트를 이동
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            table_points[dragging_point] = [x, y]

    # 마우스 버튼을 뗐을 때 포인트 이동 완료
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

        # 포인트가 수정될 때마다 테이블 깊이 값을 업데이트
        matrix = cv2.getPerspectiveTransform(table_points, transformed_points)
        transformed_depth_map = cv2.warpPerspective(depth_map, matrix, transformed_size)
        table_depth_value = np.mean(transformed_depth_map)

# 원본 이미지 창에서 마우스 이벤트 활성화
cv2.namedWindow('Original Frame')
cv2.setMouseCallback('Original Frame', mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # RGB 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 깊이 추정
    depth = depth_estimator(pil_image)
    depth_map = np.array(depth["depth"])
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Perspective Transform 적용
    matrix = cv2.getPerspectiveTransform(table_points, transformed_points)
    transformed_frame = cv2.warpPerspective(frame, matrix, transformed_size)
    transformed_depth_map = cv2.warpPerspective(depth_map, matrix, transformed_size)

    # 테이블 영역의 깊이 값 추출
    table_depth_value = np.mean(transformed_depth_map)

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

            # Perspective Transform을 적용한 후 손가락 끝의 좌표 변환
            finger_point = np.array([[finger_x, finger_y]], dtype='float32')
            finger_point = np.array([finger_point])
            transformed_finger_point = cv2.perspectiveTransform(finger_point, matrix)[0][0]
            transformed_finger_x, transformed_finger_y = int(transformed_finger_point[0]), int(transformed_finger_point[1])

            # 변환된 손가락 끝의 깊이 추출
            if 0 <= transformed_finger_x < transformed_size[0] and 0 <= transformed_finger_y < transformed_size[1]:
                finger_depth = transformed_depth_map[transformed_finger_y, transformed_finger_x]

                # 터치 판정: 테이블 영역의 깊이와 유사한지 확인 (오차 범위 10)
                if abs(finger_depth - table_depth_value) < 10:
                    cv2.putText(transformed_frame, 'Touch Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손가락 끝에 원 그리기
            cv2.circle(transformed_frame, (transformed_finger_x, transformed_finger_y), 5, (0, 255, 0), -1)

    # 원본 이미지에 테이블 모서리 표시
    for point in table_points:
        cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # 변환된 이미지와 원본 이미지 출력
    depth_colormap = cv2.applyColorMap(transformed_depth_map, cv2.COLORMAP_JET)
    cv2.imshow('Transformed Frame', transformed_frame)
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Transformed Depth Map', depth_colormap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()