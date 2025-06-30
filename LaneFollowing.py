#!/usr/bin/env python3

import os
import time
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, jsonify
from pyzbar import pyzbar
import math
import serial
import struct
import threading
from collections import deque

# --- 串口通信配置 ---
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyCH343USB0")
BAUDRATE = int(os.getenv("SERIAL_BAUD", 115200))
SERIAL_TIMEOUT = 1.0
SERIAL_RATE_HZ = 30.0 # 串口发送速度指令的频率

FRAME_HEADER = 0x7B # 数据帧头部标识
FRAME_TAIL = 0x7D   # 数据帧尾部标识

# 全局标志，用于指示是否检测到二维码
qr_code_detected = False

# 全局变量，用于存储最近10帧的角速度历史
angular_speed_history = deque(maxlen=6)

# 全局变量，用于红线计时
red_line_start_time = None

class SerialController:
    def __init__(self):
        self.ser = None
        self.last_sent = time.time()
        self.period = 1.0 / SERIAL_RATE_HZ
        self.lock = threading.Lock() # 用于线程同步，确保速度更新和发送的原子性
        self.vx = 0.0 # 线速度
        self.vy = 0.0 # 侧向速度 (对于轮式小车通常为0)
        self.vz = 0.0 # 角速度
        
    def connect(self):
        """尝试连接串口"""
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
            print(f"Serial port {SERIAL_PORT} connected successfully.")
            return True
        except Exception as e:
            print(f"Error connecting to serial port {SERIAL_PORT}: {e}")
            return False
            
    def build_frame(self, vx: float, vy: float, vz: float) -> bytes:
        """
        构建串口通信帧，包含线速度 vx, 侧向速度 vy, 角速度 vz。
        使用 struct.pack('<f', ...) 将浮点数打包为小端序的4字节浮点数。
        """
        frame = bytearray()
        frame.append(FRAME_HEADER) # 添加帧头
        frame += struct.pack('<f', vx) # 添加 vx
        frame += struct.pack('<f', vy) # 添加 vy
        frame += struct.pack('<f', vz) # 添加 vz
        frame.append(FRAME_TAIL)   # 添加帧尾
        return bytes(frame)
        
    def update_speed(self, vx: float, vy: float, vz: float):
        """
        更新内部的速度值，供 send_loop 周期性发送。
        """
        with self.lock: # 锁定以确保线程安全
            self.vx = vx
            self.vy = vy
            self.vz = vz
            
    def send_loop(self):
        """
        周期性地通过串口发送速度指令。在独立的线程中运行。
        """
        while True:
            try:
                now = time.time()
                # 控制发送频率，确保按 SERIAL_RATE_HZ 发送
                if now - self.last_sent < self.period:
                    time.sleep(max(0, self.period - (now - self.last_sent)))
                
                with self.lock: # 锁定以确保线程安全
                    frame = self.build_frame(self.vx, self.vy, self.vz)
                    if self.ser and self.ser.is_open:
                        self.ser.write(frame)
                
                self.last_sent = time.time()
            except serial.SerialException as e:
                print(f"Serial communication error: {e}. Attempting reconnect...")
                if self.ser:
                    self.ser.close()
                self.ser = None # 标记为断开连接
                time.sleep(1) # 等待一小段时间后尝试重新连接
                self.connect() 
            except Exception as e:
                print(f"Unexpected error in serial send loop: {e}. Retrying connection...")
                if self.ser:
                    self.ser.close()
                self.ser = None 
                time.sleep(1) 
                self.connect()

class PIDController:
    """
    PID 控制器实现。
    """
    def __init__(self, kp, ki, kd, max_output=float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.reset()
        
    def reset(self):
        """
        重置 PID 控制器的内部状态（积分项和上一个误差）。
        通常在目标变化或长时间不活动后调用，防止积分饱和。
        """
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()
        
    def update(self, error):
        """
        根据当前误差计算 PID 输出。
        """
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: # 避免除以零或负的 dt
            return 0.0
            
        p = self.kp * error

        self.integral += error * dt
        # 限制积分项，防止积分饱和 (wind-up)
        self.integral = max(min(self.integral, self.max_output / self.ki), -self.max_output / self.ki)
        i = self.ki * self.integral
    
        # 导数项：当前误差与上次误差的变化率
        d = self.kd * (error - self.last_error) / dt
    
        output = p + i + d
        # 限制总输出
        output = max(min(output, self.max_output), -self.max_output) 
        
        self.last_error = error
        self.last_time = now
        
        return output

# --- PID 控制器实例及其误差计算 ---

# PID 控制器实例及其参数（需要仔细调优！）
# kp: 比例增益，越大响应越快，但可能不稳定
# ki: 积分增益，越大消除静态误差的能力越强，但可能导致超调和振荡
# kd: 微分增益，越大抑制超调的能力越强，但可能对噪声敏感

# 位置 PID：主要控制小车横向位置，使其保持在车道中央
# 误差单位是像素，输出是影响角速度的因子
# 这些参数需要根据实际小车和赛道情况进行调整
position_pid = PIDController(kp=0.003, ki=0.00005, kd=0.0008, max_output= 0.4) 

# 角度 PID：主要控制小车朝向，使其与车道线方向对齐
# 误差单位是弧度，输出是影响角速度的因子
# 这些参数需要根据实际小车和赛道情况进行调整
angle_pid = PIDController(kp=0.6, ki=0.002, kd=0.05, max_output= 0.8) 

# 用于存储上次成功检测到线时的线速度和角速度
# 初始值可以设置为0，表示小车启动时静止，直到检测到线
# 如果希望小车启动就以一个默认速度前进，可以设置 linear_speed 的初始值
last_known_linear_speed = 0.0 
last_known_angular_speed = 0.0

def calculate_position_error(points, frame_width):
    """
    计算小车相对于车道线中心的侧向位置误差。
    我们假设车道线在图像中的理想中心是 frame_width // 2。
    根据约定：线在图像左边 -> 位置误差为正。
              线在图像右边 -> 位置误差为负。
    """
    if not points:
        return 0.0 # 如果没有检测到点，则没有位置误差
    # 优先使用图像底部区域的点的平均X坐标，这些点更接近小车当前位置，
    # 增加实时性和响应速度。
    # 按 Y 坐标降序排序，最底部的点在列表前面
    sorted_points_by_y = sorted(points, key=lambda p: p[1], reverse=True)
    
    # 取最底部的 1/4 点来计算平均X坐标，至少取1个点
    num_bottom_points = max(1, len(sorted_points_by_y) // 4)
    relevant_points = sorted_points_by_y[:num_bottom_points]
    
    if not relevant_points: # 再次检查以防万一
        return 0.0

    avg_x = sum(p[0] for p in relevant_points) / len(relevant_points)
    
    # 计算 (图像中心X - 检测到的线中心X)。
    # 如果 avg_x < frame_width // 2 (线在左边)，则 error > 0 (正误差)。
    # 如果 avg_x > frame_width // 2 (线在右边)，则 error < 0 (负误差)。
    error = (frame_width // 2) - avg_x
    return error

def calculate_angle_error(points):
    """
    计算车道线的角度误差，即车道线相对于图像垂直方向的夹角。
    根据约定：线向左倾斜 (需要左转) -> 角度误差为正。
              线向右倾斜 (需要右转) -> 角度误差为负。
    """
    if len(points) < 2:
        return 0.0 # 至少需要两个点才能计算角度

    # 为了获取车道线的整体方向，使用最上面和最下面的点来近似车道线方向。
    # 按 Y 坐标升序排序，top_point 是图像上方，bottom_point 是图像下方
    sorted_points = sorted(points, key=lambda p: p[1])
    if len(sorted_points) < 2: # 再次检查以防万一
        return 0.0

    top_point = sorted_points[0]
    bottom_point = sorted_points[-1]

    dx = bottom_point[0] - top_point[0] # X 轴变化量 (从上到下)
    dy = bottom_point[1] - top_point[1] # Y 轴变化量 (从上到下，通常为正)
        
    if dy == 0: # 避免除以零，如果车道线完全水平，角度误差为0
        return 0.0
            
    # math.atan2(dx, dy) 返回从正 Y 轴到点(dx, dy) 的角度。
    # dx > 0 (线向右倾斜) -> atan2 返回正值。
    # dx < 0 (线向左倾斜) -> atan2 返回负值。
    raw_angle_radians = math.atan2(dx, dy)
    
    # 根据你的约定：线向左倾斜 (dx为负) -> 角度误差应为正。
    #               线向右倾斜 (dx为正) -> 角度误差应为负。
    # 这与 atan2 的原始输出符号相反，因此需要取反。
    final_angle_error = raw_angle_radians
    
    # print(f"Raw Lane Angle (radians): {raw_angle_radians:.4f}, Adjusted Angle Error: {final_angle_error:.4f}, Degrees: {math.degrees(final_angle_error):.2f}")
    return final_angle_error

def dual_loop_control(frame, points, is_red_line_detected): # Added is_red_line_detected parameter
    """
    结合位置环和角度环的 PID 输出，计算线速度和角速度。
    根据你的要求：正误差值应该导致正角速度（左转）。
    负误差值应导致负角速度（右转）。
    如果丢线，则保持上次的线速度和角速度。
    """
    global last_known_linear_speed, last_known_angular_speed, qr_code_detected, angular_speed_history, red_line_start_time

    # New constant: Reduced speed when the general lane is recognized
    LANE_FOLLOWING_REDUCED_LINEAR_SPEED = 0.25 
    # New constant: Further reduced speed when a specific red line is detected
    RED_LINE_DETECTED_SPEED = 0.10 

    # 如果检测到二维码，则停止小车
    if qr_code_detected:
        print("QR code detected! Stopping car.")
        position_pid.reset()
        angle_pid.reset()
        angular_speed_history.clear() # Clear history on stop
        red_line_start_time = None # Reset red line timer on QR stop
        return 0.0, 0.0 # 停止线速度和角速度

    # 红线计时逻辑
    if is_red_line_detected:
        if not points:  # 只有同时没有黑线时才计时
            if red_line_start_time is None:
                red_line_start_time = time.time()  # 首次检测开始计时
                print("Red line detected without black lane. Starting 5s timer.")
                return 0.1, 0.0 # 速度很慢，直走
            elif time.time() - red_line_start_time > 5:
                print("Red line over 5s! Force stop.")
                qr_code_detected = True # 模拟检测到二维码以停止小车
                return 0.0, 0.0  # 强制停止
            else:
                # Continue slow straight while red line detected and no black line
                return 0.1, 0.0
        else:
            red_line_start_time = None  # 检测到黑线则重置计时
    else:
        red_line_start_time = None  # 未检测到红线则重置计时


    # 如果没有检测到任何车道点 (丢线)
    if not points:
        position_pid.reset() # 丢线时重置PID，避免积分饱和(reset积分项为0)
        angle_pid.reset()
        
        # New logic for lost lane
        if angular_speed_history:
            positive_angles = sum(1 for angle in angular_speed_history if angle > 0)
            negative_angles = sum(1 for angle in angular_speed_history if angle < 0)

            if positive_angles > negative_angles:
                # Most recent turns were positive (left), so try turning left
                angular_speed = 0.8
                linear_speed = 0.2 # Maintain some linear speed
                print(f"No lane points. Majority positive angles. Turning left with V={linear_speed:.2f}, R={angular_speed:.2f}")
            elif negative_angles > positive_angles:
                # Most recent turns were negative (right), so try turning right
                angular_speed = -0.8
                linear_speed = 0.2 # Maintain some linear speed
                print(f"No lane points. Majority negative angles. Turning right with V={linear_speed:.2f}, R={angular_speed:.2f}")
            else:
                # Equal positive/negative or all zeros, use last known
                linear_speed = last_known_linear_speed
                angular_speed = last_known_angular_speed
                print(f"No lane points. Equal angles or zeros. Continuing with last known speeds: V={linear_speed:.2f}, R={angular_speed:.2f}")
        else:
            # History is empty, default to last known
            linear_speed = last_known_linear_speed
            angular_speed = last_known_angular_speed
            print(f"No lane points. History empty. Continuing with last known speeds: V={linear_speed:.2f}, R={angular_speed:.2f}")

        return linear_speed, angular_speed

    # --- 成功检测到车道线时，正常计算速度 ---

    # 1. 计算位置误差和其 PID 输出
    pos_error = calculate_position_error(points, 1280)
    # print(f"Calculated Position Error: {pos_error:.2f} pixels")
    pos_output = position_pid.update(pos_error)
    
    # 2. 计算角度误差和其 PID 输出
    angle_error = calculate_angle_error(points)
    # print(f"Calculated Angle Error: {angle_4f} radians")
    angle_output = angle_pid.update(angle_error)

    # --- 融合线速度和角速度 ---
    
    # 基础前进线速度，小车在直线上应该保持的速度
    # If a specific red line is detected, further reduce the base speed
    
    base_linear_speed = 0.40 # Use the general reduced speed
    
    # 角速度（转向）的计算是关键。
    # 侧向位置误差和航向角度误差都应影响转向。
    # 权重（W_pos, W_angle）需要反复试验来找到最佳平衡点。
    W_pos = 0.5  # 位置误差对转向的影响权重
    W_angle = 0.76 # 角度误差对转向的影响权重
    
    # 根据你的要求：正的 PID 输出 (对应正误差，即线在左边或左倾斜) 导致正角速度 (左转)。
    # 负的 PID 输出 (对应负误差，即线在右边或右倾斜) 导致负角速度 (右转)。
    # 所以直接将两者加权求和作为角速度。
    angular_speed = (pos_output * W_pos) + (angle_output * W_angle)
    
    # 将角速度限制在合理范围内，通常是 PID 控制器 max_output 的限制
    angular_speed = max(min(angular_speed,5), -5)

    # 线速度调整：在急转弯时适当减速以保持稳定
    # abs(angular_speed) 越大，表示转弯越急
    speed_reduction_factor = 0.50 # 这个因子也需要调优[0.0 - 1.0]
    linear_speed = base_linear_speed - abs(angular_speed) * speed_reduction_factor
    
    # 确保线速度不为负，并且有最低速度（如果需要持续移动）
    linear_speed = max(0.0, linear_speed) # 不允许倒车，最低速度为0

    # 更新上次成功检测到线时的速度和角速度，以备丢线时使用
    last_known_linear_speed = linear_speed / 1.2
    last_known_angular_speed = angular_speed * 1.2
    angular_speed_history.append(angular_speed) # Store angular speed

    print(f"Pos PID Output: {pos_output:.4f}, Angle PID Output: {angle_output:.4f}")
    print(f"Final Linear Speed (V): {linear_speed:.2f}, Final Angular Speed (R): {angular_speed:.2f}")
    
    return linear_speed, angular_speed

# --- 摄像头和图像处理设置 ---
DEVICE = os.getenv("CAM_DEVICE", 0) # 摄像头设备索引，默认为0
TARGET_W = int(os.getenv("CAM_WIDTH", 1280)) # 目标图像宽度
TARGET_H = int(os.getenv("CAM_HEIGHT", 720)) # 目标图像高度
TARGET_FPS = int(os.getenv("CAM_FPS", 30))      # 摄像头捕获帧率
STREAM_FPS = int(os.getenv("STREAM_FPS", 15))    # Web 流传输帧率
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 70)) # JPEG 编码质量
FOURCC = "MJPG" # 摄像头视频编码格式
FRAME_INTERVAL = 1.0 / STREAM_FPS # 计算 Web 流每帧之间的时间间隔

# 初始化摄像头
cap = cv2.VideoCapture(DEVICE, cv2.CAP_V4L2)
if not cap.isOpened():
    raise RuntimeError("Failed to open camera. Please check device index, connection, and permissions.")

# 设置摄像头属性
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*FOURCC))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,   TARGET_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
cap.set(cv2.CAP_PROP_FPS,           TARGET_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # 减少缓冲区大小以降低延迟    

# 获取摄像头实际分辨率和帧率
src_w, src_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
src_fps = cap.get(cv2.CAP_PROP_FPS) or 0 # 如果获取失败，默认为0
print(f"Camera native resolution: {src_w}x{src_h} @ {src_fps:.1f} FPS")


def fit_to_canvas(frame: np.ndarray) -> np.ndarray:
    """
    调整图像大小以适应目标画布尺寸，并居中显示。
    """
    h, w = frame.shape[:2]
    scale = min(TARGET_W / w, TARGET_H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if scale != 1.0:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        frame = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.full((TARGET_H, TARGET_W, 3), 128, np.uint8) # 灰色背景
    x0, y0 = (TARGET_W - new_w)//2, (TARGET_H - new_h)//2
    canvas[y0:y0+new_h, x0:x0+new_w] = frame
    return canvas

def process_frame_hsv(frame):
    """
    使用 HSV 颜色空间进行颜色过滤，提取车道线。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 针对黑色或深色车道线进行调整。
    # H: 色相 (0-180), S: 饱和度(0-255), V: 亮度 (0-255)
    # 对于黑色，通常 V 值很低，S 值也很低。
    # 如果你的线是灰色，可能需要调整 V 值的上限。
    lower_black = np.array([0, 0, 0])      # 纯黑
    upper_black = np.array([180, 255, 80]) # 允许一些灰度，V上限80是一个经验值，可调
    mask = cv2.inRange(hsv, lower_black, upper_black) # 创建二值掩码
    kernel = np.ones((5, 5), np.uint8) # 增大核大小，提高形态学操作效果
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # 开运算：去除小点噪声(先腐蚀后膨胀)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel) # 闭运算：连接断开的小间隙 (先膨胀后腐蚀)
    mask_clean = cv2.dilate(mask_clean, kernel, iterations=1) # 膨胀：使线更粗，便于检测
    return mask_clean

def detect_red_line_presence(frame):
    """
    独立检测图像中是否存在红色线条。
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 定义红色在HSV颜色空间中的两个范围 (因为色相环是循环的)
    # 红色下限
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    # 红色上限
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # 合并两个红色掩码
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 可选的形态学操作，以去除噪声并连接红色区域
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    # 检查红色像素的数量是否超过阈值，以判断是否存在明显的红线
    # 这个阈值需要根据实际情况调整
    min_red_pixel_area = 500 # 最小红色像素面积，低于此值认为是噪声
    if np.sum(red_mask) > min_red_pixel_area:
        return True, red_mask
    return False, red_mask


def extract_lane_points(mask, num_rows=4, visualize_on=None): 
    h, w = mask.shape

    roi_start_y = h // 3 
    roi = mask[roi_start_y:, :]  
    row_height = roi.shape[0] // num_rows 
    points = []

    MIN_LANE_CONTOUR_AREA = 3500
    MAX_ASPECT_RATIO = 3.0 # 新增：最大长宽比

    for i in range(num_rows):
        y_start_in_roi = i * row_height
        y_end_in_roi = (i + 1) * row_height
        row_mask = roi[y_start_in_roi:y_end_in_roi, :]
        
        if visualize_on is not None:
            top_abs_y = y_start_in_roi + roi_start_y
            bottom_abs_y = y_end_in_roi + roi_start_y
            cv2.rectangle(visualize_on, (0, top_abs_y), (w, bottom_abs_y), (255, 255, 0), 1) 

        contours, _ = cv2.findContours(row_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        filtered_contours = [c for c in contours if cv2.contourArea(c) > MIN_LANE_CONTOUR_AREA]

        valid_contours = []
        for c in filtered_contours:
            x, y_in_row, w_box, h_box = cv2.boundingRect(c)
            # 筛选掉长比宽大于7的框
            if h_box > 0 and (w_box / h_box) <= MAX_ASPECT_RATIO:
                valid_contours.append(c)

        # Use valid_contours to get the contour with the largest area after filtering
        if valid_contours: 
            contour = max(valid_contours, key=cv2.contourArea) 
            x, y_in_row, w_box, h_box = cv2.boundingRect(contour)
            cx = x + w_box // 2
            cy_in_row = y_in_row + h_box // 2
            
            abs_cy = cy_in_row + y_start_in_roi + roi_start_y
            points.append((cx, abs_cy))

            if visualize_on is not None:
                cv2.rectangle(visualize_on, (x, y_in_row + y_start_in_roi + roi_start_y),
                                (x + w_box, y_in_row + h_box + y_start_in_roi + roi_start_y),
                                (0, 0, 255), 2)
                cv2.circle(visualize_on, (cx, abs_cy), 5, (0, 255, 0), -1)

    return points

def detect_qrcode(image): 
    """
    在给定图像中检测二维码，并绘制边界框和数据。
    """
    global qr_code_detected # 声明使用全局变量

    # 转换为灰度图以进行二维码检测
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    start_time = time.time()
    decoded_objs = pyzbar.decode(gray) # 使用 pyzbar 检测二维码
    elapsed_time = time.time() - start_time
    fps = 1.0 / elapsed_time if elapsed_time > 0 else 0

    if decoded_objs: # 如果检测到二维码
        qr_code_detected = True # 设置全局标志为True
    else: # If no QR code detected, set the flag to False
        qr_code_detected = False

    for obj in decoded_objs:
        qr_data = obj.data.decode('utf-8') # 解码二维码数据
        points = obj.polygon # 二维码的四个角点
        # 绘制多边形边界
        if len(points) > 4: # 如果点数多于4，使用凸包确保绘制正确的多边形
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        # 在二维码上方显示解码数据
        text_position = (pts[0][0][0], pts[0][0][1] - 10)
        cv2.putText(image, qr_data, text_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 在图像左上角显示二维码检测 FPS
    cv2.putText(image, f'QR FPS: {fps:.1f}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('QR Code Detection', image) # 显示二维码检测窗口
    cv2.waitKey(1) # 等待1毫秒，确保窗口响应

# --- 统计信息和 Flask 应用设置 ---
STATS = {
    "ideal_fps": TARGET_FPS,
    "cam_fps": src_fps,
    "web_fps": 0.0,
    "latency_ms": 0.0,
    "cam_resolution": f"{src_w}x{src_h}",
    "linear_speed": 0.0,
    "angular_speed": 0.0
}
_cnt, _t0 = 0, time.time() # 用于计算 Web 流 FPS
_last_sent = 0.0 # 用于控制 Web 流帧率
serial_ctrl = SerialController() # 创建串口控制器实例
app = Flask(__name__) # 初始化 Flask 应用

def gen_frames():
    """
    生成视频帧的生成器函数，用于 Flask 视频流。
    """
    global _cnt, _t0, _last_sent, qr_code_detected
    params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    
    while True:
        now = time.time()
        # 控制 Web 流的帧率
        if now - _last_sent < FRAME_INTERVAL:
            time.sleep(max(0, FRAME_INTERVAL - (now - _last_sent)))
        _last_sent = time.time()

        t_cap = time.time() # 记录帧捕获开始时间
        ok, frame = cap.read() # 从摄像头读取一帧
        if not ok:
            print("Failed to grab frame.")
            time.sleep(0.1) # 短暂等待后重试
            continue
            
        # 调整帧尺寸以适应目标画布
        frame = frame if (frame.shape[1], frame.shape[0]) == (TARGET_W, TARGET_H) else fit_to_canvas(frame)
        STATS["latency_ms"] = (time.time() - t_cap) * 1000 # 计算处理延迟
    
        # 图像处理和循线控制
        mask = process_frame_hsv(frame) # 颜色过滤得到二值掩码
        points = extract_lane_points(mask, num_rows=4, visualize_on=frame) # 提取车道线中心点
    
        # Separate red line detection
        is_red_line_detected, red_mask = detect_red_line_presence(frame)
        if is_red_line_detected:
            # Optionally display the red_mask for debugging
            cv2.imshow('Red Line Mask', red_mask)
        else:
            # Close the window if red line is not detected, or keep displaying empty mask
            try:
                cv2.destroyWindow('Red Line Mask')
            except cv2.error:
                pass


        # 检测二维码，并更新全局标志
        detect_qrcode(frame)
    
        # 计算线速度和角速度，并处理丢线情况
        linear_speed, angular_speed = dual_loop_control(frame, points, is_red_line_detected) # Pass new parameter
        print("shape :",frame.shape[1])
        # 更新全局统计数据并发送指令给串口
        STATS["linear_speed"] = linear_speed
        STATS["angular_speed"] = angular_speed
        serial_ctrl.update_speed(linear_speed, 0, angular_speed ) # vy 保持为0，因为是平面运动
        
        # 在视频流上叠加控制值(调试用)
        cv2.putText(frame, f"Speed: {linear_speed:.2f}", (10, 60),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Turn: {angular_speed:.2f}", (10, 90),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add red line detection status to visualization
        status_text = "Red Line: Detected" if is_red_line_detected else "Red Line: Not Detected"
        cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2) # Yellow color

    
        # 显示本地 OpenCV 窗口（如果不需要，可以注释掉）
        cv2.imshow("Camera Feed & Lane/QR Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): # 按'q' 键退出
            break

        # 计算网络流 FPS
        _cnt += 1
        if time.time() - _t0 >= 1: # 每秒更新一次 FPS
            STATS["web_fps"] = _cnt / (time.time() - _t0)
            _cnt, _t0 = 0, time.time()

        # 编码帧用于网络流传输
        ok, buf = cv2.imencode('.jpg', frame, params)
        if not ok:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

@app.route('/')
def index():
    """
    Flask 根路由，返回 HTML 页面显示视频流和统计信息。
    """
    html = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
    <title>MiniCar Control</title>
    <style>
      html,body{{height:100%;margin:0;overflow:hidden;background:#666;color:#fff;display:flex;flex-direction:column;font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}}
      .title{{background:#FFD700;color:#000;text-align:center;font-size:1.4rem;padding:0.6rem 0;font-weight:normal;flex-shrink:0}}
      .view{{flex:1 1 auto;display:flex;align-items:center;justify-content:center}}
      .view img{{max-width:100%;max-height:100%;object-fit:contain}}
      footer{{position:fixed;bottom:0;left:0;width:100%;background:#444;color:#eee;padding:4px 0;font-size:0.9rem;text-align:center}}
      footer span{{color:#0f0;margin:0 0.3rem}}
    </style></head><body>
      <div class='title'>MiniCar - Lane Following & QR Detection</div>
      <div class='view'><img src='{{{{ url_for('video_feed') }}}}' alt='camera'></div>
      <footer>
        Ideal FPS:<span id='ideal'>--</span> Cam FPS:<span id='cam'>--</span> Web FPS:<span id='web'>--</span> Latency:<span id='lat'>--</span>ms Resolution:<span id='res'>--</span>
      </footer>
      <script>
        async function poll(){{
          try{{ const d = await (await fetch('/stats')).json();
            document.getElementById('ideal').textContent = d.ideal_fps.toFixed(1);
            document.getElementById('cam').textContent  = d.cam_fps.toFixed(1);
            document.getElementById('web').textContent  = d.web_fps.toFixed(1);
            document.getElementById('lat').textContent  = d.latency_ms.toFixed(1);
            document.getElementById('res').textContent  = d.cam_resolution;
          }}catch(e){{console.error("Failed to fetch stats:", e);}}
          setTimeout(poll,1000);
        }} poll();
      </script></body></html>"""
    return render_template_string(html)


@app.route('/video_feed')
def video_feed():
    """
    Flask 视频流路由，返回多部分混合流。
    """
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stats')
def stats():
    """
    Flask 统计信息路由，返回 JSON 格式的当前状态。
    """
    return jsonify(STATS)


if __name__ == '__main__':
    # 尝试连接串口
    if serial_ctrl.connect():
        # 如果连接成功，启动串口发送线程(守护线程，随主程序退出而退出)
        serial_thread = threading.Thread(target=serial_ctrl.send_loop, daemon=True)
        serial_thread.start()
        print("Serial send thread started.")
    else:
        print(f"Could not connect to serial port {SERIAL_PORT}. Running without serial control.")
        # 如果串口是循线必需的，并且无法连接，可以考虑在此处退出程序：
        # import sys
        # sys.exit("Exiting: Serial connection failed.")

    print("Starting Flask web server on http://0.0.0.0:8080")
    try:
        # 启动 Flask web 服务器
        app.run('0.0.0.0', 8080, threaded=True) # threaded=True 允许多个客户端同时连接
    except Exception as e:
        print(f"Flask application error: {e}")
    finally:
        # 程序停止时进行清理工作
        cap.release() # 释放摄像头资源
        cv2.destroyAllWindows() # 关闭所有 OpenCV 窗口
        if serial_ctrl.ser and serial_ctrl.ser.is_open:
            serial_ctrl.ser.close() # 关闭串口连接
        print("Application stopped. Camera and serial port released.")
