import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 손가락이 사각형 안에 있는지 확인하는 함수
def is_finger_in_rectangle(finger_pos, top_left, bottom_right):
    finger_x, finger_y = finger_pos
    top_left_x, top_left_y = top_left
    bottom_right_x, bottom_right_y = bottom_right

    return top_left_x <= finger_x <= bottom_right_x and top_left_y <= finger_y <= bottom_right_y

# 사각형을 그리는 함수
def draw_rectangle(image, top_left, bottom_right, text, is_finger_inside):
    color = (0, 0, 255) if is_finger_inside else (0, 255, 0)  # 빨강 또는 초록
    cv2.rectangle(image, top_left, bottom_right, color, 2)

    # 사각형의 중심 좌표 계산
    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2
    
    # 텍스트 설정 및 크기 조정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # 텍스트를 사각형 중앙에 배치하기 위한 좌표 설정
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2

    # 텍스트 추가 (사각형 내부)
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return image

# 2줄에 5개씩 사각형 그리기 함수 (화면 크기에 맞춰 가변적으로 설정)
def draw_grid_of_rectangles(image, finger_pos, rows=2, cols=5):
    img_height, img_width, _ = image.shape

    # 사각형 크기 및 간격을 화면 크기에 따라 설정
    rect_width = int(0.5*img_width // (cols + 1))  # 화면 너비에 맞춰 가변적인 사각형 너비 설정
    rect_height = int(0.5*img_height // (rows + 2))  # 화면 높이에 맞춰 가변적인 사각형 높이 설정
    spacing_x = rect_width // 5 * 5  # 가로 간격
    spacing_y = rect_height // 4 * 2 # 세로 간격

    idx = 1

    for row in range(rows):
        for col in range(cols):
            # 사각형의 좌측 상단과 우측 하단 좌표 계산
            top_left_x = 100+(col + 1) * spacing_x + col * rect_width
            top_left_y = 500+(row + 1) * spacing_y + row * rect_height
            bottom_right_x = top_left_x + rect_width
            bottom_right_y = top_left_y + rect_height

            if finger_pos is not None:
                # 손가락 끝 좌표가 사각형 내부에 있는지 확인
                is_finger_inside = is_finger_in_rectangle(finger_pos, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y))

            else:
                is_finger_inside = False

            # 사각형 그리기 (손가락이 사각형 안에 있으면 색상을 변경)
            image = draw_rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), str(idx), is_finger_inside)
            idx += 1

    return image

# 화면 오른쪽 아래에 제스처를 출력하는 함수
def display_gesture(image, gesture_text):
    # 텍스트 위치: 화면 오른쪽 아래
    h, w, _ = image.shape
    position = (w - 400, h - 50)  # 오른쪽 아래로 텍스트 위치 설정
    
    # 텍스트 설정
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    color = (255, 255, 255)  # 흰색 텍스트

    # 텍스트 출력
    cv2.putText(image, f'Gesture: {gesture_text}', position, font, font_scale, color, font_thickness)

    return image

# 각도를 계산하는 함수
def calculate_angles(joint):
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]  # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]  # Child joint
    v = v2 - v1  # [20, 3]
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]  # Normalize

    angle = np.arccos(np.einsum('nt,nt->n', v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
    return np.degrees(angle)  # Convert radian to degree

# 손동작을 인식하는 함수 (PyTorch 모델 사용)
def recognize_action(model, input_data, actions, action_seq, device):
    model.eval()
    input_data = input_data.to(device)
    with torch.no_grad():
        y_pred = model(input_data).squeeze()
        i_pred = int(torch.argmax(y_pred))
        conf = torch.softmax(y_pred, dim=0)[i_pred].item()
        # print(i_pred, conf)
    if conf < 0.5:
        return None, action_seq

    action = actions[i_pred]
    action_seq.append(action)

    if len(action_seq) < 3:
        return None, action_seq

    if action_seq[-1] == action_seq[-2] == action_seq[-3]:
        return action, action_seq

    return None, action_seq

# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, top_left, bottom_right, device):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    seq = []
    action_seq = []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
    out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    wait_click = True
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        img0 = img.copy()
        img = cv2.flip(img, 1)
        result = hands.process(img)

        # draw_rectangle(img, top_left, bottom_right, 'test',False)
        img = draw_grid_of_rectangles(img, None)
        img = display_gesture(img, '')

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                ## 손가락 좌표와 각도 추출 ##
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
                ## 손가락 좌표와 각도 추출 ##

                action, action_seq = recognize_action(model, input_data, actions, action_seq, device)
                # print(f'action: {action}, action_seq: {action_seq}')
                if action is None:
                    continue

                finger_pos = (int(result.multi_hand_landmarks[0].landmark[8].x * w),
                            int(result.multi_hand_landmarks[0].landmark[8].y * h))

                if action == "click":
                    # is_finger_inside = is_finger_in_rectangle(finger_pos, top_left, bottom_right)
                    # img = draw_rectangle(img, top_left, bottom_right, 'test' ,is_finger_inside)
                    print('Wait on click')
                    if wait_click:
                        print('Click')
                        img = draw_grid_of_rectangles(img, finger_pos)
                        wait_click = False
                else:
                    wait_click = True
                img = display_gesture(img, action)

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

# 메인 함수
def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    model = torch.jit.load('models/lstm_model_scr.pt')
    model.to(device)

    actions = ['click', 'stanby1', 'stanby2']
    seq_length = 30

    top_left = (800, 500)
    bottom_right = (1200, 700)

    cap = cv2.VideoCapture(0)
    process_video(cap, model, actions, seq_length, top_left, bottom_right, device)

if __name__ == "__main__":
    main()