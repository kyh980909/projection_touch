import cv2
import mediapipe as mp
import numpy as np
import torch
from utils import *

# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, width, height, device):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # 버튼 정보 생성
    buttons = []
    rows = 3
    cols = 3
    button_texts = [f'{i+1}' for i in range(rows * cols)]
    button_texts.append('*')
    button_texts.append('0')
    button_texts.append('#')

    # 버튼 크기 및 위치 비율 설정 (화면 크기 비율에 맞춰 동적으로 설정)
    button_width_ratio = 0.1  # 버튼 너비를 화면의 10%로 설정
    button_height_ratio = 0.1  # 버튼 높이를 화면의 10%로 설정
    spacing_ratio = 0.05  # 버튼 간의 간격을 화면 비율에 맞춤
    x_offset_ratio = 0.25
    y_offset_ratio = 0.5

    key_map = {}

    for row in range(rows+1):
        for col in range(cols):
            # 버튼의 크기 및 위치 계산 (비율 기반으로 설정)
            x = round(width * x_offset_ratio + (col + 1) * spacing_ratio * width + col * button_width_ratio * width)
            y = round(height * y_offset_ratio + (row + 1) * spacing_ratio * height + row * button_height_ratio * height)
            button_size = [round(button_width_ratio * width), round(button_height_ratio * height)]
            button_text = button_texts[row * cols + col]
            buttons.append(Button(pos=[x, y], size=button_size, text=button_text))

             # 각 버튼의 인덱스와 값을 key_map에 추가
            key_map[row * cols + col + 1] = button_text  # 인덱스를 1부터 시작

    seq = []
    action_seq = []

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('input_keypad3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    out2 = cv2.VideoWriter('output_keypad3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

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

        if not wait_click:
            img = display_click_status(img, 'Wait for click', width, height, size_ratio=0.5)

        # 각 버튼들의 인덱스와 값을 화면에 출력 (draw_legend 함수 활용)
        img = draw_legend(img, key_map, width, height, size_ratio=1.0)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                # 손가락 좌표와 각도 추출
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                angle = calculate_angles(joint)
                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = torch.tensor(np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0))
                
                # 손동작 인식
                action, action_seq = recognize_action(model, input_data, actions, action_seq, device)
                if action is None:
                    continue

                finger_pos = (int(result.multi_hand_landmarks[0].landmark[8].x * width),
                            int(result.multi_hand_landmarks[0].landmark[8].y * height))

                # 손동작 인식 후 화면에 제스처 출력
                if action is not None:
                    img = display_gesture(img, action, width, height, size_ratio=0.5)

                if action == "click":
                    for button in buttons:
                        x, y = button.pos
                        w, h = button.size
                        
                        if is_finger_in_rectangle(finger_pos, button):
                            if wait_click:
                                text = button.text
                                # 화면 밖으로 나가지 않도록 좌표 보정
                                rect_x1 = max(0, x - w // 2)
                                rect_y1 = max(0, y - h // 2)
                                rect_x2 = min(width, x + w // 2)
                                rect_y2 = min(height, y + h // 2)
                                
                                # 사각형 그리기
                                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), thickness=2)
                                wait_click = False
                else:
                    wait_click = True
                # img = display_gesture(img, action, width=w, height=h)

                # 검지 손가락 끝에 원 그리기
                cv2.circle(img, finger_pos, 5, (255, 0, 0), -1)

                # 텍스트 출력
                cv2.putText(img, f'{action.upper()}', org=(finger_pos[0], finger_pos[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        out.write(img0)
        out2.write(img)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    model = torch.jit.load('models/lstm_model_scr.pt')
    model.to(device)

    actions = ['click', 'wait', 'grib']
    seq_length = 30

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'width: {width}, height: {height}')
    process_video(cap, model, actions, seq_length, width, height, device)