import cv2
import mediapipe as mp
import numpy as np
import torch
import subprocess
from utils import *
import socket #소켓 통신
import threading
import queue
from models.model import *

client_socket = None

#사용자 입력을 받는 함수
def get_user_input(input_queue):
    while True:
        user_input = input("Enter a command: ")
        input_queue.put(user_input)
        if user_input == 'q':
            break

#서버에 연결하는 함수
def connect_to_server():
    global client_socket #전역변수로 선언
    HOST = 'beaglebone.local'  # 서버의 IP 주소
    PORT = 8888  # 서버의 포트 번호

    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        print("서버에 연결되었습니다.")
    except Exception as e:
        print(f"서버 연결 실패: {e}")
        client_socket = None

#서버에 텍스트를 전송하는 함수
def send_text(text):
    global client_socket
    if client_socket:
        try:
            client_socket.sendall(text.encode('utf-8'))
            print("텍스트 전송 완료")
        except Exception as e:
            print(f"전송 중 오류 발생: {e}")
            # 연결이 끊어졌을 경우 재연결 시도
            connect_to_server()
    else:
        print("서버에 연결되어 있지 않습니다.")

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

def projection_area_auto_detection(cap):
    #사용자 입력을 받는 스레드 생성
    input_queue = queue.Queue()

    input_thread = threading.Thread(target=get_user_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()
    corners = []
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break
        
        result_corners, result_image = find_green_corners(img)
        if result_corners:
            corners = result_corners

        try:
            user_input = input_queue.get_nowait()
            if user_input.lower() == 'q':
                break
            send_text(user_input) #사용자 입력 서버에 전송
            print(f"User input: {user_input}") #사용자 입력 확인
        except queue.Empty:
            pass

        cv2.imshow('img', result_image)  # ROI가 그려진 이미지만 'img'에 표시
        if cv2.waitKey(1) == ord('q'):
            print(corners)
            break
        
    return corners

def handle_gesture_actions(img, action, buttons, width, height, finger_pos, pers, wait_click, wait_open_setting):
    transformed_finger_pos = convert_position(finger_pos, pers)

    if action is not None:
        for button in buttons:
            x, y = button.pos
            w, h = button.size

            if action == "click":
                if is_finger_in_rectangle(transformed_finger_pos, button):
                    rect_x1 = max(0, x - w // 2)
                    rect_y1 = max(0, y - h // 2)
                    rect_x2 = min(width, x + w // 2)
                    rect_y2 = min(height, y + h // 2)
                    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), thickness=2)
                    
                    if wait_click:
                        text = button.text
                        send_text(text)
                        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), thickness=2)
                        wait_click = False
                        wait_open_setting = True

            elif action == "setting":
                if wait_open_setting:
                    subprocess.Popen(["gnome-control-center", "display"])
                    wait_open_setting = False
                    wait_click = True

            else:
                wait_click = True
                wait_open_setting = True
    else:
        wait_click = True
        wait_open_setting = True

    cv2.circle(img, finger_pos, 5, (255, 0, 0), -1)
    return wait_click, wait_open_setting

# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, width, height, device, corners):

    global srcQuad, dragSrc, ptOld, img

    dragSrc= [False, False, False, False]

    # 모서리 점들의 좌표, 드래그 상태 여부
    srcQuad = np.array([corners[0], corners[2], corners[3], corners[1]], np.float32)
    dstQuad = np.array([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]], np.float32)
    
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    # 버튼 정보 생성
    buttons = []
    rows = 2
    cols = 5
    button_texts = [f'{i+1}' for i in range(rows * cols)]
    button_texts[-1] = button_texts[-1].replace('10', '0')

    # 버튼 크기 및 위치 비율 설정 (화면 크기 비율에 맞춰 동적으로 설정)
    button_width_ratio = 0.1  # 버튼 너비를 화면의 10%로 설정
    button_height_ratio = 0.1  # 버튼 높이를 화면의 10%로 설정
    spacing_ratio = 0.1  # 버튼 간의 간격을 화면 비율에 맞춤
    x_offset_ratio = 0.05
    y_offset_ratio = 0.05

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

    seq = []
    action_seq = []

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))
    out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', width, height)
    cv2.setMouseCallback('img', onMouse)

    cv2.namedWindow('dst', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst', width, height)

    #사용자 입력을 받는 스레드 생성
    input_queue = queue.Queue()

    input_thread = threading.Thread(target=get_user_input, args=(input_queue,))
    input_thread.daemon = True
    input_thread.start()

    wait_click = True
    wait_open_setting = True
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to capture frame from camera. Exiting.")
            break

        img0 = img.copy()
        img = draw(img, buttons, width, height, size_ratio=1.0, position_ratio=1.0)
        result = hands.process(img0)

        try:
            user_input = input_queue.get_nowait()
            if user_input.lower() == 'q':
                break
            send_text(user_input) #사용자 입력 서버에 전송
            print(f"User input: {user_input}") #사용자 입력 확인
        except queue.Empty:
            pass

        if not wait_click:
            img = display_click_status(img, 'Wait for click', width, height, size_ratio=0.25)
        else:
            img = display_click_status(img, '', width, height, size_ratio=0.25)

        # 각 버튼들의 인덱스와 값을 화면에 출력 (draw_legend 함수 활용)
        img = draw_legend(img, key_map, width, height, size_ratio=1.0)

        # 모서리점, 사각형 그리기 (img에만 적용)
        img_with_roi = drawROI(img, srcQuad, 1.0)
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
                
                transformed_finger_pos = convert_position(finger_pos, pers)

                wait_click, wait_open_setting = handle_gesture_actions(
                    img, action, buttons, width, height, finger_pos, pers, wait_click, wait_open_setting
                )

        out.write(img0)
        out2.write(img)
        cv2.imshow('img', img)  # ROI가 그려진 이미지만 'img'에 표시
        if cv2.waitKey(1) == ord('q'):
            break

    if client_socket:
        client_socket.close()

    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #서버에 연결
    connect_to_server()
    device = torch.device('cuda')

    # Hyperparameters
    features = 99
    num_classes = 4

    model = LSTMModel(input_size=features, hidden_size=64, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('models/gesture_model.pth'))
    model.to(device)

    actions = ['click', 'wait', 'power on', 'power off']
    seq_length = 30

    cap = cv2.VideoCapture(0)
    width = 640
    height = 480
    print(f'width: {width}, height: {height}')
    corners = projection_area_auto_detection(cap)
    process_video(cap, model, actions, seq_length, width, height, device, corners)