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
def draw_rectangle(image, top_left, bottom_right, is_finger_inside):
    color = (0, 0, 255) if is_finger_inside else (0, 255, 0)  # 빨강 또는 초록
    cv2.rectangle(image, top_left, bottom_right, color, 2)
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
        print(i_pred, conf)
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
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    seq = []
    action_seq = []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
    out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        img0 = img.copy()
        img = cv2.flip(img, 1)
        result = hands.process(img)

        draw_rectangle(img, top_left, bottom_right, False)

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
                print(f'action: {action}, action_seq: {action_seq}')
                if action is None:
                    continue

                finger_pos = (int(result.multi_hand_landmarks[0].landmark[8].x * w),
                            int(result.multi_hand_landmarks[0].landmark[8].y * h))

                if action == "click":
                    is_finger_inside = is_finger_in_rectangle(finger_pos, top_left, bottom_right)
                    img = draw_rectangle(img, top_left, bottom_right, is_finger_inside)

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
    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    
    # model = LSTMModel(input_size=99, hidden_size=64, num_classes=3)  # 적절한 input_size 및 num_classes 설정
    # model.load_state_dict(torch.load('models/lstm_model.pth'))  # 모델 불러오기
    
    model = torch.jit.load('lstm_model_scr.pt')
    model.to(device)

    actions = ['click', 'stanby1', 'stanby2']
    seq_length = 30

    top_left = (800, 500)
    bottom_right = (1200, 700)

    cap = cv2.VideoCapture(0)
    process_video(cap, model, actions, seq_length, top_left, bottom_right, device)

if __name__ == "__main__":
    main()