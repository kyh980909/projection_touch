import cv2
import numpy as np
import time
import subprocess
import logging
from control import *
import datetime
from Constants import *
import os
import fcntl
import mmap
import struct
import array
import socket
import json
import threading
import signal
import sys

running = True

def signal_handler(sig, frame):
    global running
    running = False
    print("Signal received, setting running to False")

signal.signal(signal.SIGINT, signal_handler)
class Button:
    def __init__(self, pos, size, text):
        self.pos = pos  # [x, y] center position
        self.size = size  # [width, height] size
        self.text = text  # text to display on the button

# active keys
active_keys = set([])
active_keys_lock = threading.Lock()

green_screen_mode = False
green_screen_lock = threading.Lock()

cv2.ocl.setUseOpenCL(False)

def create_buttons(width, height):
    buttons = []
    rows = 2
    cols = 5
    button_texts = ['{0}'.format(i+1) for i in range(rows * cols)]
    button_texts[-1] = button_texts[-1].replace('10', '0')

    # Ratio settings
    button_width_ratio = 0.1  # button width 10%
    button_height_ratio = 0.1  # button height 10%
    spacing_ratio = 0.1  # button space
    x_offset_ratio = 0
    y_offset_ratio = 0.1

    key_map = {}

    for row in range(rows):
        for col in range(cols):
            x = round(width * x_offset_ratio + (col + 1) * spacing_ratio * width + col * button_width_ratio * width)
            y = round(height * y_offset_ratio + (row + 1) * spacing_ratio * height + row * button_height_ratio * height)
            button_size = [round(button_width_ratio * width), round(button_height_ratio * height)]
            button_text = button_texts[row * cols + col]
            buttons.append(Button(pos=[x, y], size=button_size, text=button_text))
            key_map[row * cols + col + 1] = button_text

    return buttons

def draw(img, buttons, width=720, height=480, size_ratio=1.0, position_ratio=1.0, flip=True):
    global active_keys 

    for button in buttons:
        x, y = button.pos
        w, h = button.size
        
        w = int(w * size_ratio)
        h = int(h * size_ratio)
        x = int(x * position_ratio)
        y = int(y * position_ratio)
        
        is_active = button.text in active_keys
        color = (255, 0, 0) if is_active else (0, 255, 0)

        cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), (0, 0, 0), -3)
        cv2.rectangle(img, (x - w//2, y - h//2), (x + w//2, y + h//2), color, -1)
        
        text_size = cv2.getTextSize(button.text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        text_color = (0, 255, 0) if is_active else (255, 0, 0)
        cv2.putText(img, button.text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

        print("Drawing button {0}, is_active: {1}".format(button.text, is_active))

    if flip:
        img = cv2.flip(img, -1)

    return img

def draw_rectangle(image, top_left, bottom_right, text, is_active=False):
    color = (0, 0, 255) if is_active else (0, 255, 0)  
    cv2.rectangle(image, top_left, bottom_right, color, 2)
    
    cv2.circle(image, top_left, 5, color, -1)
    cv2.circle(image, bottom_right, 5, color, -1)
    cv2.circle(image, (top_left[0], bottom_right[1]), 5, color, -1)
    cv2.circle(image, (bottom_right[0], top_left[1]), 5, color, -1)

    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    text_x = int(center_x - text_size[0] // 2)
    text_y = int(center_y + text_size[1] // 2)

    if is_active:
        text_color = (255, 255, 255)  
    else:
        text_color = (0, 0, 0)  

    cv2.putText(image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return image

def draw_grid_of_rectangles(image, rows=2, cols=5):
    global active_keys
    img_height, img_width, _ = image.shape

    rect_width = int(img_width // (cols + 1))
    rect_height = int(img_height // (rows + 2))
    spacing_x = rect_width // 5
    spacing_y = rect_height // 4

    idx = 1

    for row in range(rows):
        for col in range(cols):
            top_left_x = (col + 1) * spacing_x + col * rect_width
            top_left_y = (row + 1) * spacing_y + row * rect_height
            bottom_right_x = top_left_x + rect_width
            bottom_right_y = top_left_y + rect_height

            is_active = str(idx) in active_keys
            print("Drawing rectangle {0}, is_active: {1}".format(idx, is_active))  
            image = draw_rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), str(idx), is_active)
            idx += 1
    return image

def handle_client(conn, addr):
    global active_keys, green_screen_mode, running
    print("{0} connected".format(addr))
    while running:
        try:
            data = conn.recv(1024)
            if not data:
                break
            text = data.decode().strip()
            if text == 'c': 
                with green_screen_lock:
                    green_screen_mode = not green_screen_mode
                print("Green screen mode: {}".format("OFF" if green_screen_mode else "ON"))
            else:
                with active_keys_lock:
                    before = set(active_keys)
                    if text in active_keys:
                        active_keys.remove(text)
                    else:
                        active_keys.add(text)
                    after = set(active_keys)
                print("Active keys before: {0}".format(before))
                print("Active keys after: {0}".format(after))
                print("Changed key: {0}".format(text))
            conn.send("OK".encode())
        except Exception as e:
            print("Client error: {0}".format(e))
            break
    conn.close()

def start_tcp_server():
    global running
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 8888))
    server.listen(5)
    server.settimeout(1) #set timeout to 1 second
    print("TCP server started on port 8888")
    
    while running:
        try:
            conn, addr = server.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.start()
        except socket.timeout:
            continue
        except Exception as e:
            print("TCP server error: {0}".format(e))
            break
    server.close()
    print("TCP server closed")

def add_click_text(image):
    img_height, img_width, _ = image.shape
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text = "Click"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = (img_width - text_size[0]) // 2  
    text_y = img_height - 20  
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return image

def run_i2c_commands():
    commands = [
        "i2cset -y 2 0x1b 0x0b 0x00 0x00 0x00 0x00 i",
        "i2cset -y 2 0x1b 0x0c 0x00 0x00 0x00 0x09 i"
    ]
    for cmd in commands:
        try:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode == 0:
                print("Successfully executed: {}".format(cmd))
            else:
                print("Error executing command: {}".format(cmd))
                print("Error details: {}".format(stderr))
        except Exception as e:
            print("Exception while executing command: {}".format(cmd))
            print("Exception details: {}".format(str(e)))

def write_to_framebuffer(image):
    try:
        with open('/dev/fb0', 'r+b') as f:
            screen_info = array.array('H', [0] * 32)
            fcntl.ioctl(f.fileno(), 0x4600, screen_info)
            xres, yres, xres_virtual, yres_virtual, bits_per_pixel = screen_info[:5]

            fix_info = array.array('c', ' ' * 68)
            fcntl.ioctl(f.fileno(), 0x4602, fix_info)
            line_length = struct.unpack('I', fix_info[32:36])[0]

            if xres <= 0 or yres <= 0:
                xres, yres = 720, 480

            if image.shape[2] == 3:  # If image is BGR
                image = cv2.cvtColor(image, cv2.COLOR_BGR2BGR565)
            
            if image.shape[:2] != (yres, xres):
                image = cv2.resize(image, (xres, yres))
            
            f.write(image.tostring())
            f.flush()
            print("Frame successfully written to framebuffer")

    except Exception as e:
        print("Error writing to framebuffer: {}".format(str(e)))

def initialize_display():
    print("Initializing display...")
    run_i2c_commands()
    time.sleep(1)
    DPP2607_Write_SystemReset()
    time.sleep(2)
    DPP2607_Write_VideoSourceSelection(SourceSel.EXTERNAL_VIDEO_PARALLEL_I_F_)
    DPP2607_Write_VideoPixelFormat(RGB888_24_BIT)
    DPP2607_Write_VideoResolution(Resolution.NHD_LANDSCAPE)
    time.sleep(1)
    run_i2c_commands()

def draw_border_and_markers(image):
    height, width = image.shape[:2]
    border_color = (255, 0, 0)  
    marker_color = (0, 255, 0) 
    thickness = 2
    marker_size = 20

    cv2.rectangle(image, (0, 0), (width-1, height-1), border_color, thickness)

    cv2.rectangle(image, (0, 0), (marker_size, marker_size), marker_color, -1)
    cv2.rectangle(image, (width-marker_size, 0), (width, marker_size), marker_color, -1)
    cv2.rectangle(image, (0, height-marker_size), (marker_size, height), marker_color, -1)
    cv2.rectangle(image, (width-marker_size, height-marker_size), (width, height), marker_color, -1)

    return image

def initialize_framebuffer():
    try:
        with open('/dev/fb0', 'wb') as f:
            black_frame = np.zeros((480, 720, 3), dtype=np.uint8)
            f.write(black_frame.tostring())
            f.flush()
        print("Framebuffer initialized")
    except Exception as e:
        print("Error initializing framebuffer: {}".format(str(e)))

def main():
    global running
    Test_name = 'OpenCV DLP2000 Keyboard Test'
    
    datalog = DataLog(LogDir, Test_name)

    logging.getLogger().setLevel(logging.DEBUG)  
    print("Opening DLP2000...")
    DPP2607_Open()
    print("Setting slave address...")
    DPP2607_SetSlaveAddr(SlaveAddr)
    print("Setting IO debug...")
    DPP2607_SetIODebug(IODebug)

    print("Initializing DLP2000...")
    time.sleep(5) 

    try:
        initialize_display()
        initialize_framebuffer()
        
        width, height = 720, 480
        frame_count = 0
        duration = 3600
        buttons = create_buttons(width, height)
        tcp_thread = threading.Thread(target=start_tcp_server)
        tcp_thread.start()

        start_time = time.time()
        last_update_time = start_time
        while running and (time.time() - start_time < duration):
            current_time = time.time()
            if current_time - last_update_time >= 0.016:  
                with green_screen_lock:
                    if green_screen_mode:
                        image = np.zeros((height, width, 3), dtype=np.uint8)
                        with active_keys_lock:
                            frame = draw(image, buttons, width, height, flip=False)
                        frame = draw_border_and_markers(frame)
                    else:
                        frame = np.full((height, width, 3), (0, 255, 0), dtype=np.uint8)

                write_to_framebuffer(frame)
                print("Screen updated, frame count: {0}, active keys: {1}, green screen mode: {2}".format(
                    frame_count, active_keys, "ON" if green_screen_mode else "OFF"))

                last_update_time = current_time
                frame_count += 1

                if frame_count % 3600 == 0:  
                    cv2.imwrite('dlp2000_keyboard_output_{}.png'.format(frame_count//3600), frame)
                    print("Image saved at {} minutes.".format(frame_count//3600))

            time.sleep(0.001)

        cv2.imwrite('dlp2000_keyboard_output_final.png', frame)
        print("Final image has been saved.")
        
        result = "OpenCV keyboard images displayed for 1 hour and saved periodically. (Pass/Fail/Stop)"
        print("Display result: {}".format(result))
        
        datalog.add_col('Test name', Test_name)
        datalog.add_col('End Time', ' ' + str(datetime.datetime.now()))
        datalog.add_col('Result', result)
        datalog.add_col('P/F Result', "Pass" if "Pass" in result else "Fail")
        datalog.log()
    except Exception as e:
        print("Test failed Exception: {}".format(str(e)))
        datalogConstants(datalog)
        datalog.add_col('Test name', Test_name)
        datalog.add_col('End Time', ' ' + str(datetime.datetime.now()))
        datalog.add_col('Result', "Test Fail EXCEPTION")        
        datalog.add_col('P/F Result', "Fail")
        datalog.log()
    finally:
        running = False
        print("Cleaning up...")
        if 'tcp_thread' in locals() and tcp_thread.is_alive():
            tcp_thread.join(timeout=5)
        run_i2c_commands()
        DPP2607_Close()
        datalog.close()
        print("Cleanup completed")

if __name__ == "__main__":
    main()
