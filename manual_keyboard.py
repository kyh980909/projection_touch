import cv2
import torch
import numpy as np
import pandas as pd
import math
import mediapipe as mp
from pynput.keyboard import Controller
from screeninfo import get_monitors
import tensorflow as tf
import pyautogui
# from keras.saving import load_model
from transformers import pipeline
from PIL import Image

device = "cpu" #"mps" if torch.backends.mps.is_available() else "cpu"

# 경량화 모델
checkpoint = "LiheYoung/depth-anything-small-hf"
depth_estimator = pipeline("depth-estimation", model=checkpoint, device=device)

class Store():
    def __init__(self,pos,size,text):
        self.pos=pos
        self.size=size
        self.text=text

def draw(img, storedVar):
    for button in storedVar:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), thickness=2)
        cv2.putText(img, button.text, (x-15, y+15), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    return img

def draw_legend(img):
    overlay = img.copy()
    output = img.copy()
    cv2.rectangle(overlay, (round(round(10/1280*width)), round(10/960*height)), (round(round(280/1280*width)), round(240/960*height)), (0, 0, 0), -1)
    i = 0
    for k, v in key_map.items():
        cv2.putText(overlay, f'{k} : {v}', (20, 40+(i*35)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        i+=1

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def draw_input(img, text):
    overlay = img.copy()
    output = img.copy()

    cv2.rectangle(overlay, (round(990/1280*width), round(900/960*height)), (round(1270/1280*width), round(950/960*height)), (0, 0, 0), -1)
    cv2.putText(overlay, text, (round(1010/1280*width), round(935/960*height)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
    return output

def vertical_symmetry(x, y, height):
    """
    세로축 (Y축) 대칭 변환
    :param x: 원본 X 좌표
    :param y: 원본 Y 좌표
    :param height: 대칭시킬 기준 선의 위치 (이미지의 높이)
    :return: 세로축에 대해 대칭된 새로운 (x, y) 좌표
    """
    new_y = height - y
    return x, new_y

def horizontal_symmetry(x, y, width):
    """
    가로축 (X축) 대칭 변환
    :param x: 원본 X 좌표
    :param y: 원본 Y 좌표
    :param width: 대칭시킬 기준 선의 위치 (이미지의 너비)
    :return: 가로축에 대해 대칭된 새로운 (x, y) 좌표
    """
    new_x = width - x
    return new_x, y

def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp

def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

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

                cpy = drawROI(src, srcQuad)
                cv2.imshow(window_name, cpy)
                ptOld = (x, y)
                break

def convert_position(pt1, pt2, pers):
    # 변환된 동차 좌표 계산
    transformed_pt1 = np.dot(pers, pt1)
    transformed_pt2 = np.dot(pers, pt2)

    # 변환된 유클리드 좌표 계산 (동차좌표를 유클리드 좌표로 변환)
    transformed_pt1 = transformed_pt1 / transformed_pt1[2]
    transformed_x1, transformed_y1 = round(transformed_pt1[0]), round(transformed_pt1[1])

    # 변환된 유클리드 좌표 계산 (동차좌표를 유클리드 좌표로 변환)
    transformed_pt2 = transformed_pt2 / transformed_pt2[2]
    transformed_x2, transformed_y2 = round(transformed_pt2[0]), round(transformed_pt2[1]) 

    return (transformed_x1, transformed_y1), (transformed_x2, transformed_y2)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

'''
model = load_model("keyboard_model1.keras") #TFSMLayer("keyboard_model1", call_endpoint='serving_default')

print("가상 인터페이스 생성중...")

# 데이터를 읽고 변환하는 과정은 그대로 유지
test_data = pd.read_hdf('1280x960.h5', 'df')
test_input = tf.convert_to_tensor(test_data, dtype=tf.float64)

# 예측 및 배열 변환
pred = model.predict(test_input)
pred = np.argmax(pred, axis=1)
pred = tf.reshape(pred, [1280, 960]).numpy()

# 벡터화된 방식으로 test 배열 생성
test = np.zeros((960, 1280, 3), dtype=np.uint8)

# np.repeat을 사용하여 예측값을 3채널로 확장
test = np.repeat(pred[:, :, np.newaxis], 3, axis=2)
'''

key_map = {"1":"1", "2":"2", "3":"3", "4":"LEFT", "5":"OK", "6":"RIGHT"}

keyboard = Controller()

# 마우스 콜백 함수를 위한 변수
points = []
transform_ready = False

monitor1 = get_monitors()[1]
monitor2 = get_monitors()[2]

window_name = "Monitor"
# window_name2 = "Projector"

cv2.namedWindow(window_name, cv2.WND_PROP_AUTOSIZE)
cv2.moveWindow(window_name, monitor1.x - 1, monitor1.y - 1)

# cv2.namedWindow(window_name2, cv2.WND_PROP_FULLSCREEN)
# cv2.moveWindow(window_name2, monitor2.x - 1, monitor2.y - 1)
# cv2.setWindowProperty(window_name2, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# 웹캠으로부터 입력 받기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

'''
가상 키패드 그리는 부분
'''
StoredVar = []
# 한 개의 임의의 키 설정
center_button = Store([640, 480], [50, 50], 'OK')  # 중앙에 사각형 버튼 하나를 생성
StoredVar.append(center_button)
img = np.full((960, 1280, 3), 0, np.uint8)
img = cv2.rectangle(img, (0, 0), (1279, 959), (255, 255, 255), 5)

'''
# 임의의 입력 데이터를 모델에 넣어 출력 텐서를 정의
sample_input = np.random.random((1, model.input_shape[1]))  # 임의의 입력 데이터 생성
output = model(sample_input)  # 모델을 호출하여 출력 텐서를 정의

# 140~150번 라인의 수정된 코드
unique_indices = np.unique(pred)  # pred의 고유 값(클래스)만 추출

# 각 클래스의 최소/최대 좌표를 한 번에 계산
for index in unique_indices:
    if index <= 1:
        continue  # 0, 1 클래스는 배경과 라인이기 때문에 제외함

    # 벡터화된 연산으로 최소/최대 좌표 계산
    indices = np.where(pred == index)
    min_y, max_y = np.min(indices[0]), np.max(indices[0])
    min_x, max_x = np.min(indices[1]), np.max(indices[1])

    StoredVar.append(Store([round(((min_x + max_x) / 2) / 1280 * width), 
                            round(((min_y + max_y) / 2) / 960 * height)], 
                           [round(((max_x - min_x) / 2) / 1280 * width), 
                            round(((max_y - min_y) / 2) / 960 * height)], 
                           str(index)))
'''
flag = 0
text = ''

img = draw(img, StoredVar)
img = draw_legend(img)
img = draw_input(img, text)

projector_img = img.copy()

mpHands = mp.solutions.hands  
Hands = mpHands.Hands(min_detection_confidence=0.4, min_tracking_confidence=0.4, max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

# 체크: 웹캠이 제대로 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

dw = 1280
dh = 960

# 입력 영상 크기 및 출력 영상 크기
h, w = 960, 1280

# 모서리 점들의 좌표, 드래그 상태 여부
srcQuad = np.array([[30, 30], [30, h-30], [w-30, h-30], [w-30, 30]], np.float32)
dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
dragSrc = [False, False, False, False]

# 가상 키보드 이미지 출력창
# cv2.imshow(window_name2, projector_img)

# 테이블 영역의 깊이 값 추출을 위한 플래그
table_depth_captured = False
table_depth_value = 0

while True:
    # 웹캠으로부터 프레임을 읽어옴
    ret, src = cap.read()
    
    # 프레임 읽기 실패 시 종료
    if not ret:
        print("Failed to grab frame")
        break

    # RGB 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

    # 깊이 추정   
    depth = depth_estimator(pil_image)
    depth_map = np.array(depth["depth"])
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

    # 테이블 영역 좌표 설정 (srcQuad 좌표를 기반으로 최소/최대 X, Y 좌표를 찾음)
    x_coords = srcQuad[:, 0]
    y_coords = srcQuad[:, 1]

    # 최소/최대 X, Y 좌표 계산
    x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
    y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
    table_depth_area = depth_map[y_min:y_max, x_min:x_max]
    table_depth_value = np.mean(table_depth_area)
    table_depth_captured = True

    # MediaPipe를 이용한 핸드 스켈레톤 추출
    rgb_frame = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    results = Hands.process(rgb_frame)

    # 모서리점, 사각형 그리기
    disp = drawROI(src, srcQuad)
    projector_img = img.copy()

    cv2.setMouseCallback(window_name, onMouse)

    # 투시 변환
    pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
    dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)

    # 테이블 영역의 깊이 값 추출
    table_depth_value = np.mean(dst)
    
    # dst = cv2.flip(dst, 1)
    # dst = cv2.flip(dst, 0)
    cv2.imshow('dst', dst)

    # 검지 손가락 끝의 좌표 및 깊이 정보 추출
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = src.shape

            # 검지 손가락 끝의 좌표 (Landmark ID 8)
            index_finger_tip = hand_landmarks.landmark[8]
            finger_x, finger_y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # 손가락 끝이 테이블 영역 내에 있는지 확인
            if (x_min <= finger_x <= x_max and
                y_min <= finger_y <= y_max):
                
                # 검지 손가락 끝의 깊이 추출
                finger_depth = depth_map[finger_y, finger_x]

                # 터치 판정: 테이블 영역의 깊이와 유사한지 확인 (예: 오차 범위 10)
                if abs(finger_depth - table_depth_value) < 10:
                    cv2.putText(src, 'Touch Detected', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 손가락 끝에 원 그리기
            cv2.circle(src, (finger_x, finger_y), 5, (0, 255, 0), -1)

    disp = draw(disp, StoredVar)
    disp = draw_legend(disp)
    disp = draw_input(disp, text)
    projector_img = draw_input(projector_img, text)
    
    cv2.imshow(window_name, disp)
    cv2.imshow('depth', depth_map)
    # cv2.imshow(window_name2, projector_img)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()