import cv2
import mediapipe as mp
import numpy as np
import torch
from utils import *
import socket #소켓 통신
import threading
import queue

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

        # corners, result_image = extract_projection_area(img)
        # corners, result_image = detect_white_corners(img)
        
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

# 키패드 버튼을 직접 생성하는 함수
def create_keypad_direct(width: int, height: int) -> list[Button]:
    buttons = []

    # 각 버튼의 위치와 크기를 직접 지정
    buttons.append(Button(pos=[round(0.3 * width), round(0.4 * height)], size=[round(0.15 * width), round(0.1 * height)], text="Button 1"))
    buttons.append(Button(pos=[round(0.7 * width), round(0.4 * height)], size=[round(0.15 * width), round(0.1 * height)], text="Button 2"))
    buttons.append(Button(pos=[round(0.5 * width), round(0.50 * height)], size=[round(0.15 * width), round(0.1 * height)], text="Button 3"))
    buttons.append(Button(pos=[round(0.25 * width), round(0.75 * height)], size=[round(0.15 * width), round(0.1 * height)], text="Button 4"))
    buttons.append(Button(pos=[round(0.5 * width), round(0.65 * height)], size=[round(0.17 * width), round(0.1 * height)], text="Button 5"))
    buttons.append(Button(pos=[round(0.75 * width), round(0.75 * height)], size=[round(0.15 * width), round(0.1 * height)], text="Button 6"))

    return buttons

# 비디오 처리 루프 함수
def process_video(cap, model, actions, seq_length, width, height, device, corners):
    
    global srcQuad, dragSrc, ptOld, img

    dragSrc= [False, False, False, False]

    # 모서리 점들의 좌표, 드래그 상태 여부
    srcQuad = np.array([corners[0], corners[2], corners[3], corners[1]], np.float32)
    dstQuad = np.array([[0, 0], [0, height-1], [width-1, height-1], [width-1, 0]], np.float32)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    # 버튼 정보 생성
    buttons = create_keypad_direct(width, height)  # 화면 크기에 맞춰 버튼 배열 생성

    key_map = {"1":"1", "2":"2", "3":"3", "4":"LEFT", "5":"OK", "6":"RIGHT"}

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
        result = hands.process(img)

        try:
            user_input = input_queue.get_nowait()
            if user_input.lower() == 'q':
                break
            send_text(user_input) #사용자 입력 서버에 전송
            print(f"User input: {user_input}") #사용자 입력 확인
        except queue.Empty:
            pass

        if not wait_click:
            img = display_click_status(img, 'Wait for click', width, height, size_ratio=0.5)

        # 각 버튼들의 인덱스와 값을 화면에 출력 (draw_legend 함수 활용)
        img = draw_legend(img, key_map, width, height, size_ratio=1.0)

        # 모서리점, 사각형 그리기 (img에만 적용)
        img_with_roi = drawROI(img, srcQuad, 1.0)
        projector_img = img_with_roi.copy()
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
                                send_text(text) #서버에 텍스트 전송
                                # 화면 밖으로 나가지 않도록 좌표 보정
                                rect_x1 = max(0, x - w // 2)
                                rect_y1 = max(0, y - h // 2)
                                rect_x2 = min(width, x + w // 2)
                                rect_y2 = min(height, y + h // 2)
                                
                                # 사각형 그리기
                                cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), thickness=2)
                                wait_click = False
                elif action == "grib":
                    if wait_open_setting:
                        # 시스템 환경설정 열기
                        subprocess.run(["gnome-control-center", "display"])
                        wait_open_setting = False

                else:
                    wait_click = True
                    wait_open_setting = True

                # 검지 손가락 끝에 원 그리기
                cv2.circle(img, finger_pos, 5, (255, 0, 0), -1)

                # 텍스트 출력
                cv2.putText(img, f'{action.upper()}', org=(finger_pos[0], finger_pos[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


        # img_with_roi = draw(img_with_roi, buttons)
        # img_with_roi = draw_legend(img_with_roi)
        # img_with_roi = draw_input(img_with_roi, text)
        # projector_img = draw_input(projector_img, text)

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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        
    model = torch.jit.load('models/lstm_model_scr2.pt')
    model.to(device)

    actions = ['click', 'wait', 'grib']
    seq_length = 30

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'width: {width}, height: {height}')
    corners = projection_area_auto_detection(cap)
    process_video(cap, model, actions, seq_length, width, height, device, corners)