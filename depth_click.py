import cv2
import mediapipe as mp
import numpy as np
import torch
from utils import *
from transformers import pipeline
from PIL import Image

# 비디오 처리 루프 함수
def process_video(cap, depth_model, width, height):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # 버튼 정보 생성
    buttons = []
    rows = 2
    cols = 5
    button_texts = [f'Button {i+1}' for i in range(rows * cols)]
    button_texts[-1] = button_texts[-1].replace('10', '0')

    # 버튼 크기 및 위치 비율 설정 (화면 크기 비율에 맞춰 동적으로 설정)
    button_width_ratio = 0.1  # 버튼 너비를 화면의 10%로 설정
    button_height_ratio = 0.1  # 버튼 높이를 화면의 10%로 설정
    spacing_ratio = 0.05  # 버튼 간의 간격을 화면 비율에 맞춤
    x_offset_ratio = 0.25
    y_offset_ratio = 0.5

    key_map = {}

    for row in range(rows):
        for col in range(cols):
            # 버튼의 크기 및 위치 계산 (비율 기반으로 설정)
            x = round(width * x_offset_ratio + (col + 1) * spacing_ratio * width + col * button_width_ratio * width)
            y = round(height * y_offset_ratio + (row + 1) * spacing_ratio * height + row * button_height_ratio * height)
            button_size = [round(button_width_ratio * width), round(button_height_ratio * height)]
            button_text = button_texts[row * cols + col]
            buttons.append(Button(pos=[x, y], size=button_size, text=button_text))

            # 각 버튼의 인덱스와 값을 key_map에 추가
        key_map[row * cols + col + 1] = button_text  # 인덱스를 1부터 시작

    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('input_keypad3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    # out2 = cv2.VideoWriter('output_keypad3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    wait_click = True
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        img0 = img.copy()
        img = cv2.flip(img, 1)
        img = draw(img, buttons, width, height, size_ratio=1.0, position_ratio=1.0)
        result = hands.process(img)

        # RGB 이미지로 변환
        pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 깊이 추정   
        depth = depth_model(pil_image)
        depth_map = np.array(depth["depth"])
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if not wait_click:
            img = display_click_status(img, 'Wait for click', width, height, size_ratio=0.5)

        # 각 버튼들의 인덱스와 값을 화면에 출력 (draw_legend 함수 활용)
        img = draw_legend(img, key_map, width, height, size_ratio=1.0)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                finger_pos = (int(result.multi_hand_landmarks[0].landmark[8].x * width),
                            int(result.multi_hand_landmarks[0].landmark[8].y * height))
                cv2.circle(img, finger_pos, 10, (0, 255, 0), -1)

                for button in buttons:
                    x, y = button.pos
                    w, h = button.size

                    if is_finger_in_rectangle(finger_pos, button):
                        finger_depth = depth_map[finger_pos[1], finger_pos[0]]
                        table_depth_area = depth_map[y-height:y+height, x-width:x+width]
                        table_depth_value = np.mean(table_depth_area)

                        # 터치 판정: 테이블 영역의 깊이와 유사한지 확인 (예: 오차 범위 10)
                        print(f'finger_depth: {finger_depth}')
                        print(f'table_depth_value: {table_depth_value}')
                        if abs(finger_depth - table_depth_value) < 50:
                            if wait_click:
                                text = button.text
                        
                                # 화면 밖으로 나가지 않도록 좌표 보정
                                rect_x1 = max(0, x - w // 2)
                                rect_y1 = max(0, y - h // 2)
                                rect_x2 = min(width, x + w // 2)
                                rect_y2 = min(height, y + h // 2)        
                        
                                # 사각형 그리기
                                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), thickness=4)
                                wait_click = False
                    else:
                        wait_click = True

        # out.write(img0)
        # out2.write(img)
        cv2.imshow('img', img)
        cv2.imshow('depth', depth_map)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    # out.release()
    # out2.release()
    cv2.destroyAllWindows()

# 메인 함수
if __name__ == "__main__":
    device = 'cpu' #torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    checkpoint = "LiheYoung/depth-anything-small-hf"
    depth_model = pipeline("depth-estimation", model=checkpoint, device=device)

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process_video(cap, depth_model, width, height)