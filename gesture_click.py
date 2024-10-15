import cv2
import mediapipe as mp
import numpy as np
import torch
from utils import *

def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, img

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25: # type: ignore
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(img, srcQuad, 1.0)
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break

# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, width, height, device):

    global srcQuad, dragSrc, ptOld, img

    dragSrc= [False, False, False, False]

    # 모서리 점들의 좌표, 드래그 상태 여부
    srcQuad = np.array([[30, 30], [30, height-30], [width-30, height-30], [width-30, 30]], np.float32)
    dstQuad = np.array([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]], np.float32)
    
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

    key_map = {}

    for row in range(rows):
        for col in range(cols):
            # 버튼의 크기 및 위치 계산 (비율 기반으로 설정)
            x = 300+round((col + 1) * spacing_ratio * width + col * button_width_ratio * width)
            y = 600+round((row + 1) * spacing_ratio * height + row * button_height_ratio * height)
            button_size = [round(button_width_ratio * width), round(button_height_ratio * height)]
            button_text = button_texts[row * cols + col]
            buttons.append(Button(pos=[x, y], size=button_size, text=button_text))

             # 각 버튼의 인덱스와 값을 key_map에 추가
            key_map[row * cols + col + 1] = button_text  # 인덱스를 1부터 시작

    seq = []
    action_seq = []

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    wait_click = True
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        img0 = img.copy()
        img = draw(img, buttons, width, height, size_ratio=1.0, position_ratio=1.0)
        result = hands.process(img)

        if not wait_click:
            img = display_click_status(img, 'Wait for click', width, height, size_ratio=0.5)

        # 각 버튼들의 인덱스와 값을 화면에 출력 (draw_legend 함수 활용)
        img = draw_legend(img, key_map, width, height, size_ratio=1.0)

        # 모서리점, 사각형 그리기 (img에만 적용)
        # img_with_roi = drawROI(img, srcQuad, 1.0)
        # projector_img = img_with_roi.copy()

        cv2.setMouseCallback('img', onMouse)

        # 투시 변환 (dst에서는 ROI와 관련된 내용 제외)
        pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
        dst = cv2.warpPerspective(img0, pers, (width, height), flags=cv2.INTER_CUBIC)  # img0를 사용하여 srcQuad 및 ROI 무시

        cv2.imshow('dst', dst)

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
                        
                        # 어떤 버튼을 클릭했는지 판단하는 조건문
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

                # 검지 손가락 끝에 원 그리기
                cv2.circle(img, finger_pos, 5, (255, 0, 0), -1)

                # 텍스트 출력
                cv2.putText(img, f'{action.upper()}', org=(finger_pos[0], finger_pos[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

        out.write(img0)
        out2.write(img)
        cv2.imshow('img', img)  # ROI가 그려진 이미지만 'img'에 표시
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    model = torch.jit.load('models/lstm_model_scr2.pt')
    model.to(device)

    actions = ['click', 'wait', 'grib']
    seq_length = 30

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'width: {width}, height: {height}')
    process_video(cap, model, actions, seq_length, width, height, device)