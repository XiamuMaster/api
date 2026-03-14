import sys
import os
import threading
import time
import cv2
import numpy as np
from datetime import datetime
import uuid
import subprocess
import requests
import torch

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QTabWidget, QMessageBox, QGroupBox,
                             QComboBox, QSlider, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget

from ultralytics import YOLO
from PIL import ImageGrab, Image

# 全局变量
flask_thread_yolo = None
flask_thread_carnum = None
model_yolo = None
model_carnum = None

# 检测设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 检测线程 - 通用
class DetectionThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)
    error = pyqtSignal(str)

    def __init__(self, mode, model_type, file_path=None, conf=0.25, webcam_frame=None):
        super().__init__()
        self.mode = mode  # 'image', 'video', 'webcam', 'screen'
        self.model_type = model_type  # 'yolo' or 'carnum'
        self.file_path = file_path
        self.conf = conf
        self.webcam_frame = webcam_frame

    def run(self):
        global model_yolo, model_carnum
        model = model_yolo if self.model_type == 'yolo' else model_carnum
        model_name = "物品" if self.model_type == 'yolo' else "车牌"
        
        try:
            if self.mode == 'image':
                self.progress.emit(f"正在检测{model_name}图片...")
                results = model(self.file_path, conf=self.conf)
                result_img = results[0].plot()

                detections = []
                if len(results[0].boxes) > 0:
                    for i in range(len(results[0].boxes)):
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                        conf_val = results[0].boxes.conf[i].cpu().numpy()
                        cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                        cls_name = model.names[cls_id]
                        detections.append({
                            'class': cls_name,
                            'confidence': float(conf_val),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })

                result_path = self.file_path.replace('.', '_annotated.')
                cv2.imwrite(result_path, result_img)

                self.finished.emit(result_path, detections)

            elif self.mode == 'video':
                self.progress.emit(f"正在检测{model_name}视频，请稍候...")
                cap = cv2.VideoCapture(self.file_path)

                video_fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if video_fps == 0:
                    video_fps = 30

                base, ext = os.path.splitext(self.file_path)
                result_path = f"{base}_annotated{ext}"
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
                out = cv2.VideoWriter(result_path, fourcc, video_fps, (width, height))
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    result_path = f"{base}_annotated.avi"
                    out = cv2.VideoWriter(result_path, fourcc, video_fps, (width, height))

                all_detections = []
                frame_count = 0
                
                # 计时变量
                start_time = time.time()
                detect_interval = max(1, int(video_fps / 5))

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model(frame, conf=self.conf)

                    if len(results[0].boxes) > 0:
                        for i in range(len(results[0].boxes)):
                            x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                            conf_val = results[0].boxes.conf[i].cpu().numpy()
                            cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                            cls_name = model.names[cls_id]
                            all_detections.append({
                                'class': cls_name,
                                'confidence': float(conf_val),
                                'bbox': [float(x1), float(y1), float(x2), float(y2)]
                            })

                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    frame_count += 1

                    if frame_count % 30 == 0:
                        elapsed = time.time() - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        self.progress.emit(f"已处理 {frame_count} 帧... 当前速度: {current_fps:.1f} FPS")

                cap.release()
                out.release()
                
                # 计算总耗时和平均 FPS
                total_time = time.time() - start_time
                avg_fps = frame_count / total_time if total_time > 0 else 0
                self.progress.emit(f"视频检测完成! 总帧数: {frame_count}, 总耗时: {total_time:.2f}秒, 平均速度: {avg_fps:.1f} FPS")

                self.finished.emit(result_path, all_detections)

            elif self.mode == 'webcam':
                self.progress.emit(f"正在检测{model_name}摄像头画面...")
                results = model(self.webcam_frame, conf=self.conf)
                result_img = results[0].plot()

                detections = []
                if len(results[0].boxes) > 0:
                    for i in range(len(results[0].boxes)):
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                        conf_val = results[0].boxes.conf[i].cpu().numpy()
                        cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                        cls_name = model.names[cls_id]
                        detections.append({
                            'class': cls_name,
                            'confidence': float(conf_val),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })

                self.finished.emit("", detections)

            elif self.mode == 'screen':
                self.progress.emit(f"正在检测{model_name}屏幕...")
                results = model(self.webcam_frame, conf=self.conf)
                result_img = results[0].plot()

                detections = []
                if len(results[0].boxes) > 0:
                    for i in range(len(results[0].boxes)):
                        x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                        conf_val = results[0].boxes.conf[i].cpu().numpy()
                        cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                        cls_name = model.names[cls_id]
                        detections.append({
                            'class': cls_name,
                            'confidence': float(conf_val),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })

                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                h, w, ch = result_rgb.shape
                qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)

                self.finished.emit("", detections)

        except Exception as e:
            self.error.emit(str(e))


# Flask 线程
class FlaskThread(QThread):
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    is_already_running = False

    def __init__(self, port=5000, model_type='yolo'):
        super().__init__()
        self.port = port
        self.model_type = model_type

    def run(self):
        try:
            cmd = [sys.executable, 'app.py', str(self.port), self.model_type]
            subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            
            time.sleep(2)
            
            for _ in range(10):
                try:
                    resp = requests.get(f'http://127.0.0.1:{self.port}/api/health', timeout=1)
                    if resp.status_code == 200:
                        break
                except:
                    pass
                time.sleep(1)
            
            FlaskThread.is_already_running = True
            self.log_signal.emit(f"{self.model_type} 服务已启动: http://127.0.0.1:{self.port}")

        except Exception as e:
            error_msg = str(e)
            self.log_signal.emit(f"启动失败: {error_msg}")
            self.error_signal.emit(error_msg)


# 主窗口
class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.detection_thread = None
        
        # 物品检测相关
        self.webcam_timer = None
        self.screen_timer = None
        self.webcam_capture = None
        self.is_webcam_running = False
        self.is_screen_running = False
        # FPS 计数器
        self.webcam_frame_count = 0
        self.webcam_last_time = time.time()
        self.screen_frame_count = 0
        self.screen_last_time = time.time()
        
        # 车牌检测相关
        self.carnum_webcam_timer = None
        self.carnum_screen_timer = None
        self.carnum_webcam_capture = None
        self.carnum_webcam_frame_count = 0
        self.carnum_webcam_last_time = time.time()
        self.carnum_screen_frame_count = 0
        self.carnum_screen_last_time = time.time()
        self.is_carnum_webcam_running = False
        self.is_carnum_screen_running = False

    def initUI(self):
        self.setWindowTitle('YOLO 智能识别系统')
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        self.btn_start_flask = QPushButton("启动服务")
        self.btn_start_flask.clicked.connect(self.start_flask)
        control_layout.addWidget(self.btn_start_flask)

        control_layout.addWidget(QLabel("置信度:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.setMaximumWidth(200)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v / 100:.2f}"))
        control_layout.addWidget(self.conf_slider)
        control_layout.addWidget(self.conf_label)

        control_group.setLayout(control_layout)
        main_layout.addWidget(control_group)

        # 主标签页：普通检测 | 车牌检测
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ===== 普通检测标签页 =====
        self.tab_normal = QWidget()
        self.tabs.addTab(self.tab_normal, "普通检测")
        self.init_normal_tab()

        # ===== 车牌检测标签页 =====
        self.tab_carnum = QWidget()
        self.tabs.addTab(self.tab_carnum, "车牌检测")
        self.init_carnum_tab()

        # ===== 日志标签页 =====
        self.tab_log = QWidget()
        self.tabs.addTab(self.tab_log, "运行日志")
        self.init_log_tab()

        self.statusBar().showMessage("就绪")

    def init_normal_tab(self):
        """普通检测 - 包含图片、视频、摄像头、屏幕四个子标签"""
        layout = QVBoxLayout()
        
        # 子标签页
        self.normal_tabs = QTabWidget()
        
        # 图片检测
        self.tab_normal_image = QWidget()
        self.init_normal_image_tab()
        self.normal_tabs.addTab(self.tab_normal_image, "图片检测")
        
        # 视频检测
        self.tab_normal_video = QWidget()
        self.init_normal_video_tab()
        self.normal_tabs.addTab(self.tab_normal_video, "视频检测")
        
        # 摄像头检测
        self.tab_normal_webcam = QWidget()
        self.init_normal_webcam_tab()
        self.normal_tabs.addTab(self.tab_normal_webcam, "摄像头检测")
        
        # 屏幕检测
        self.tab_normal_screen = QWidget()
        self.init_normal_screen_tab()
        self.normal_tabs.addTab(self.tab_normal_screen, "屏幕检测")
        
        layout.addWidget(self.normal_tabs)
        self.tab_normal.setLayout(layout)

    def init_normal_image_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal_image = QPushButton("选择图片")
        self.btn_normal_image.clicked.connect(lambda: self.select_file('image', 'yolo'))
        btn_layout.addWidget(self.btn_normal_image)
        layout.addLayout(btn_layout)
        
        self.normal_image_label = QLabel("请选择图片进行检测")
        self.normal_image_label.setAlignment(Qt.AlignCenter)
        self.normal_image_label.setMinimumHeight(400)
        self.normal_image_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.normal_image_label)
        
        self.normal_image_result = QTextEdit()
        self.normal_image_result.setMaximumHeight(100)
        self.normal_image_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.normal_image_result)
        
        self.normal_image_progress = QProgressBar()
        self.normal_image_progress.setVisible(False)
        layout.addWidget(self.normal_image_progress)
        
        self.tab_normal_image.setLayout(layout)

    def init_normal_video_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal_video = QPushButton("选择视频")
        self.btn_normal_video.clicked.connect(lambda: self.select_file('video', 'yolo'))
        btn_layout.addWidget(self.btn_normal_video)
        layout.addLayout(btn_layout)
        
        # 视频播放器
        self.normal_video_widget = QVideoWidget()
        self.normal_video_widget.setMinimumHeight(400)
        self.normal_video_widget.setStyleSheet("border: 1px solid #ccc; background: #000;")
        layout.addWidget(self.normal_video_widget)
        
        self.normal_video_player = QMediaPlayer()
        self.normal_video_player.setVideoOutput(self.normal_video_widget)
        
        self.normal_video_result = QTextEdit()
        self.normal_video_result.setMaximumHeight(100)
        self.normal_video_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.normal_video_result)
        
        self.normal_video_progress = QProgressBar()
        self.normal_video_progress.setVisible(False)
        layout.addWidget(self.normal_video_progress)
        
        self.tab_normal_video.setLayout(layout)

    def init_normal_webcam_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal_webcam = QPushButton("开启摄像头")
        self.btn_normal_webcam.clicked.connect(self.toggle_normal_webcam)
        btn_layout.addWidget(self.btn_normal_webcam)
        layout.addLayout(btn_layout)
        
        self.normal_webcam_label = QLabel("点击按钮开启摄像头")
        self.normal_webcam_label.setAlignment(Qt.AlignCenter)
        self.normal_webcam_label.setMinimumHeight(400)
        self.normal_webcam_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.normal_webcam_label)
        
        self.normal_webcam_result = QTextEdit()
        self.normal_webcam_result.setMaximumHeight(100)
        self.normal_webcam_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.normal_webcam_result)
        
        self.tab_normal_webcam.setLayout(layout)

    def init_normal_screen_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_normal_screen = QPushButton("开启屏幕检测")
        self.btn_normal_screen.clicked.connect(self.toggle_normal_screen)
        btn_layout.addWidget(self.btn_normal_screen)
        
        self.btn_normal_capture = QPushButton("截取屏幕")
        self.btn_normal_capture.clicked.connect(self.capture_normal_screen)
        btn_layout.addWidget(self.btn_normal_capture)
        layout.addLayout(btn_layout)
        
        self.normal_screen_label = QLabel("点击按钮开始屏幕检测")
        self.normal_screen_label.setAlignment(Qt.AlignCenter)
        self.normal_screen_label.setMinimumHeight(400)
        self.normal_screen_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.normal_screen_label)
        
        self.normal_screen_result = QTextEdit()
        self.normal_screen_result.setMaximumHeight(100)
        self.normal_screen_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.normal_screen_result)
        
        self.tab_normal_screen.setLayout(layout)

    def init_carnum_tab(self):
        """车牌检测 - 包含图片、视频、摄像头、屏幕四个子标签"""
        layout = QVBoxLayout()
        
        # 子标签页
        self.carnum_tabs = QTabWidget()
        
        # 图片检测
        self.tab_carnum_image = QWidget()
        self.init_carnum_image_tab()
        self.carnum_tabs.addTab(self.tab_carnum_image, "图片检测")
        
        # 视频检测
        self.tab_carnum_video = QWidget()
        self.init_carnum_video_tab()
        self.carnum_tabs.addTab(self.tab_carnum_video, "视频检测")
        
        # 摄像头检测
        self.tab_carnum_webcam = QWidget()
        self.init_carnum_webcam_tab()
        self.carnum_tabs.addTab(self.tab_carnum_webcam, "摄像头检测")
        
        # 屏幕检测
        self.tab_carnum_screen = QWidget()
        self.init_carnum_screen_tab()
        self.carnum_tabs.addTab(self.tab_carnum_screen, "屏幕检测")
        
        layout.addWidget(self.carnum_tabs)
        self.tab_carnum.setLayout(layout)

    def init_carnum_image_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_carnum_image = QPushButton("选择图片")
        self.btn_carnum_image.clicked.connect(lambda: self.select_file('image', 'carnum'))
        btn_layout.addWidget(self.btn_carnum_image)
        layout.addLayout(btn_layout)
        
        self.carnum_image_label = QLabel("请选择图片进行检测")
        self.carnum_image_label.setAlignment(Qt.AlignCenter)
        self.carnum_image_label.setMinimumHeight(400)
        self.carnum_image_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.carnum_image_label)
        
        self.carnum_image_result = QTextEdit()
        self.carnum_image_result.setMaximumHeight(100)
        self.carnum_image_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.carnum_image_result)
        
        self.carnum_image_progress = QProgressBar()
        self.carnum_image_progress.setVisible(False)
        layout.addWidget(self.carnum_image_progress)
        
        self.tab_carnum_image.setLayout(layout)

    def init_carnum_video_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_carnum_video = QPushButton("选择视频")
        self.btn_carnum_video.clicked.connect(lambda: self.select_file('video', 'carnum'))
        btn_layout.addWidget(self.btn_carnum_video)
        layout.addLayout(btn_layout)
        
        # 视频播放器
        self.carnum_video_widget = QVideoWidget()
        self.carnum_video_widget.setMinimumHeight(400)
        self.carnum_video_widget.setStyleSheet("border: 1px solid #ccc; background: #000;")
        layout.addWidget(self.carnum_video_widget)
        
        self.carnum_video_player = QMediaPlayer()
        self.carnum_video_player.setVideoOutput(self.carnum_video_widget)
        
        self.carnum_video_result = QTextEdit()
        self.carnum_video_result.setMaximumHeight(100)
        self.carnum_video_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.carnum_video_result)
        
        self.carnum_video_progress = QProgressBar()
        self.carnum_video_progress.setVisible(False)
        layout.addWidget(self.carnum_video_progress)
        
        self.tab_carnum_video.setLayout(layout)

    def init_carnum_webcam_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_carnum_webcam = QPushButton("开启摄像头")
        self.btn_carnum_webcam.clicked.connect(self.toggle_carnum_webcam)
        btn_layout.addWidget(self.btn_carnum_webcam)
        layout.addLayout(btn_layout)
        
        self.carnum_webcam_label = QLabel("点击按钮开启摄像头")
        self.carnum_webcam_label.setAlignment(Qt.AlignCenter)
        self.carnum_webcam_label.setMinimumHeight(400)
        self.carnum_webcam_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.carnum_webcam_label)
        
        self.carnum_webcam_result = QTextEdit()
        self.carnum_webcam_result.setMaximumHeight(100)
        self.carnum_webcam_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.carnum_webcam_result)
        
        self.tab_carnum_webcam.setLayout(layout)

    def init_carnum_screen_tab(self):
        layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_carnum_screen = QPushButton("开启屏幕检测")
        self.btn_carnum_screen.clicked.connect(self.toggle_carnum_screen)
        btn_layout.addWidget(self.btn_carnum_screen)
        
        self.btn_carnum_capture = QPushButton("截取屏幕")
        self.btn_carnum_capture.clicked.connect(self.capture_carnum_screen)
        btn_layout.addWidget(self.btn_carnum_capture)
        layout.addLayout(btn_layout)
        
        self.carnum_screen_label = QLabel("点击按钮开始屏幕检测")
        self.carnum_screen_label.setAlignment(Qt.AlignCenter)
        self.carnum_screen_label.setMinimumHeight(400)
        self.carnum_screen_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.carnum_screen_label)
        
        self.carnum_screen_result = QTextEdit()
        self.carnum_screen_result.setMaximumHeight(100)
        self.carnum_screen_result.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.carnum_screen_result)
        
        self.tab_carnum_screen.setLayout(layout)

    def init_log_tab(self):
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(btn_clear)

        self.tab_log.setLayout(layout)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def start_flask(self):
        global flask_thread_yolo, flask_thread_carnum, model_yolo, model_carnum

        if flask_thread_yolo and flask_thread_yolo.is_already_running:
            self.log("服务已在运行中")
            return

        try:
            # 显示设备信息
            if torch.cuda.is_available():
                self.log(f"使用 GPU 加速: {torch.cuda.get_device_name(0)}")
            else:
                self.log("使用 CPU 推理")

            self.log("正在加载 YOLO 模型...")
            model_yolo = YOLO('yolo26x.pt')
            model_yolo.to(DEVICE)  # 使用 GPU 加速
            self.log("物品检测模型加载成功!")

            self.log("正在加载车牌模型...")
            model_carnum = YOLO('yolo_carnum_best.pt')
            model_carnum.to(DEVICE)  # 使用 GPU 加速
            self.log("车牌检测模型加载成功!")

            flask_thread_yolo = FlaskThread(5000, 'yolo')
            flask_thread_yolo.log_signal.connect(self.log)
            flask_thread_yolo.start()

            flask_thread_carnum = FlaskThread(5001, 'carnum')
            flask_thread_carnum.log_signal.connect(self.log)
            flask_thread_carnum.start()

            self.btn_start_flask.setEnabled(False)
            self.btn_start_flask.setText("服务运行中")

        except Exception as e:
            self.log(f"启动失败: {e}")
            QMessageBox.critical(self, "错误", f"启动失败: {e}")

    def select_file(self, file_type, detect_type):
        global model_yolo, model_carnum

        model = model_yolo if detect_type == 'yolo' else model_carnum
        if model is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        if file_type == 'image':
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv)"
            )

        if not file_path:
            return

        conf = self.conf_slider.value() / 100
        self.log(f"开始检测: {os.path.basename(file_path)}")

        # 根据检测类型和文件类型设置对应的控件
        if detect_type == 'yolo':
            if file_type == 'image':
                progress = self.normal_image_progress
                progress.setVisible(True)
            else:
                progress = self.normal_video_progress
                progress.setVisible(True)
            progress.setRange(0, 0)
            
            self.detection_thread = DetectionThread(file_type, 'yolo', file_path, conf)
            self.detection_thread.progress.connect(lambda m: self.log(m))
            self.detection_thread.finished.connect(lambda path, dets: self.on_image_finished(path, dets, 'yolo', file_type))
            self.detection_thread.error.connect(lambda e: self.on_detection_error(e))
            self.detection_thread.start()
        else:
            if file_type == 'image':
                progress = self.carnum_image_progress
                progress.setVisible(True)
            else:
                progress = self.carnum_video_progress
                progress.setVisible(True)
            progress.setRange(0, 0)
            
            self.detection_thread = DetectionThread(file_type, 'carnum', file_path, conf)
            self.detection_thread.progress.connect(lambda m: self.log(m))
            self.detection_thread.finished.connect(lambda path, dets: self.on_image_finished(path, dets, 'carnum', file_type))
            self.detection_thread.error.connect(lambda e: self.on_detection_error(e))
            self.detection_thread.start()

    def on_image_finished(self, result_path, detections, detect_type, file_type):
        self.log(f"检测完成! 发现 {len(detections)} 个目标")

        result_msg = f"检测到 {len(detections)} 个目标:\n"
        for i, det in enumerate(detections[:10]):
            result_msg += f"{i + 1}. {det['class']} - {det['confidence']:.2f}\n"
        if len(detections) > 10:
            result_msg += f"... 还有 {len(detections) - 10} 个目标"

        # 根据检测类型和文件类型显示结果
        if detect_type == 'yolo':
            if file_type == 'image':
                self.normal_image_progress.setVisible(False)
                self.normal_image_result.setText(result_msg)
                if result_path and os.path.exists(result_path):
                    pixmap = QPixmap(result_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            self.normal_image_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.normal_image_label.setPixmap(scaled_pixmap)
            else:
                self.normal_video_progress.setVisible(False)
                self.normal_video_result.setText(result_msg)
                if result_path and os.path.exists(result_path):
                    self.log(f"正在播放视频: {result_path}")
                    self.normal_video_player.stop()
                    media_content = QMediaContent(QUrl.fromLocalFile(os.path.abspath(result_path)))
                    self.normal_video_player.setMedia(media_content)
                    self.normal_video_player.play()
        else:
            if file_type == 'image':
                self.carnum_image_progress.setVisible(False)
                self.carnum_image_result.setText(result_msg)
                if result_path and os.path.exists(result_path):
                    pixmap = QPixmap(result_path)
                    if not pixmap.isNull():
                        scaled_pixmap = pixmap.scaled(
                            self.carnum_image_label.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        )
                        self.carnum_image_label.setPixmap(scaled_pixmap)
            else:
                self.carnum_video_progress.setVisible(False)
                self.carnum_video_result.setText(result_msg)
                if result_path and os.path.exists(result_path):
                    self.log(f"正在播放视频: {result_path}")
                    self.carnum_video_player.stop()
                    media_content = QMediaContent(QUrl.fromLocalFile(os.path.abspath(result_path)))
                    self.carnum_video_player.setMedia(media_content)
                    self.carnum_video_player.play()

    def on_detection_error(self, error):
        self.normal_image_progress.setVisible(False)
        self.normal_video_progress.setVisible(False)
        self.carnum_image_progress.setVisible(False)
        self.carnum_video_progress.setVisible(False)
        self.log(f"检测错误: {error}")
        QMessageBox.critical(self, "错误", f"检测失败: {error}")

    # ===== 普通检测 - 摄像头 =====
    def toggle_normal_webcam(self):
        global model_yolo

        if model_yolo is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        if self.is_webcam_running:
            self.is_webcam_running = False
            if self.webcam_capture:
                self.webcam_capture.release()
            if self.webcam_timer:
                self.webcam_timer.stop()
            self.btn_normal_webcam.setText("开启摄像头")
            self.normal_webcam_label.setText("摄像头已关闭")
            self.log("普通检测摄像头已停止")
        else:
            self.webcam_capture = cv2.VideoCapture(0)
            if not self.webcam_capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return

            # 重置 FPS 计数器
            self.webcam_frame_count = 0
            self.webcam_last_time = time.time()
            
            self.is_webcam_running = True
            self.btn_normal_webcam.setText("关闭摄像头")
            self.log("普通检测摄像头已开启")

            self.webcam_timer = QTimer()
            self.webcam_timer.timeout.connect(self.update_normal_webcam)
            self.webcam_timer.start(30)

    def update_normal_webcam(self):
        if not self.is_webcam_running or not self.webcam_capture:
            return

        ret, frame = self.webcam_capture.read()
        if not ret:
            return

        frame_copy = frame.copy()
        conf = self.conf_slider.value() / 100
        results = model_yolo(frame_copy, conf=conf)
        result_img = results[0].plot()

        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.normal_webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.normal_webcam_label.setPixmap(scaled_pixmap)

        # 计算 FPS
        self.webcam_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.webcam_last_time
        fps = 0
        if elapsed >= 1.0:
            fps = self.webcam_frame_count / elapsed
            self.webcam_frame_count = 0
            self.webcam_last_time = current_time
            # 立即显示 FPS
            self.normal_webcam_result.setText(f"未检测到目标\n检测速度: {fps:.1f} FPS")
            return

        # 检测结果
        detections = []
        if len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                conf_val = results[0].boxes.conf[i].cpu().numpy()
                cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                cls_name = model_yolo.names[cls_id]
                detections.append(f"{cls_name} ({conf_val:.2f})")
        
        result_text = "\n".join(detections) if detections else "未检测到目标"
        self.normal_webcam_result.setText(result_text)

    # ===== 普通检测 - 屏幕 =====
    def toggle_normal_screen(self):
        global model_yolo

        if model_yolo is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        if self.is_screen_running:
            self.is_screen_running = False
            if self.screen_timer:
                self.screen_timer.stop()
            self.btn_normal_screen.setText("开启屏幕检测")
            self.log("普通检测屏幕检测已停止")
        else:
            # 重置 FPS 计数器
            self.screen_frame_count = 0
            self.screen_last_time = time.time()
            
            self.is_screen_running = True
            self.btn_normal_screen.setText("停止屏幕检测")
            self.log("普通检测屏幕检测已开启")

            self.screen_timer = QTimer()
            self.screen_timer.timeout.connect(self.update_normal_screen)
            self.screen_timer.start(100)

    def update_normal_screen(self):
        if not self.is_screen_running:
            return

        try:
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            conf = self.conf_slider.value() / 100
            results = model_yolo(frame, conf=conf)
            result_img = results[0].plot()

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.normal_screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.normal_screen_label.setPixmap(scaled_pixmap)

            # 计算 FPS
            self.screen_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.screen_last_time
            fps = 0
            if elapsed >= 1.0:
                fps = self.screen_frame_count / elapsed
                self.screen_frame_count = 0
                self.screen_last_time = current_time
                self.normal_screen_result.setText(f"未检测到目标\n检测速度: {fps:.1f} FPS")
                return

            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model_yolo.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")
            
            result_text = "\n".join(detections) if detections else "未检测到目标"
            self.normal_screen_result.setText(result_text)

        except Exception as e:
            self.log(f"普通检测屏幕检测错误: {e}")

    def capture_normal_screen(self):
        global model_yolo

        if model_yolo is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        try:
            self.log("正在截取屏幕...")
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            conf = self.conf_slider.value() / 100
            results = model_yolo(frame, conf=conf)
            result_img = results[0].plot()

            result_path = f"screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.jpg"
            cv2.imwrite(result_path, result_img)

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.normal_screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.normal_screen_label.setPixmap(scaled_pixmap)

            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model_yolo.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")
            self.normal_screen_result.setText("\n".join(detections) if detections else "未检测到目标")

            self.log(f"屏幕截取完成: {result_path}")

        except Exception as e:
            self.log(f"屏幕截取错误: {e}")
            QMessageBox.critical(self, "错误", f"屏幕截取失败: {e}")

    # ===== 车牌检测 - 摄像头 =====
    def toggle_carnum_webcam(self):
        global model_carnum

        if model_carnum is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        if self.is_carnum_webcam_running:
            self.is_carnum_webcam_running = False
            if self.carnum_webcam_capture:
                self.carnum_webcam_capture.release()
            if self.carnum_webcam_timer:
                self.carnum_webcam_timer.stop()
            self.btn_carnum_webcam.setText("开启摄像头")
            self.carnum_webcam_label.setText("摄像头已关闭")
            self.log("车牌检测摄像头已停止")
        else:
            self.carnum_webcam_capture = cv2.VideoCapture(0)
            if not self.carnum_webcam_capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return

            # 重置 FPS 计数器
            self.carnum_webcam_frame_count = 0
            self.carnum_webcam_last_time = time.time()
            
            self.is_carnum_webcam_running = True
            self.btn_carnum_webcam.setText("关闭摄像头")
            self.log("车牌检测摄像头已开启")

            self.carnum_webcam_timer = QTimer()
            self.carnum_webcam_timer.timeout.connect(self.update_carnum_webcam)
            self.carnum_webcam_timer.start(30)

    def update_carnum_webcam(self):
        if not self.is_carnum_webcam_running or not self.carnum_webcam_capture:
            return

        ret, frame = self.carnum_webcam_capture.read()
        if not ret:
            return

        frame_copy = frame.copy()
        conf = self.conf_slider.value() / 100
        results = model_carnum(frame_copy, conf=conf)
        result_img = results[0].plot()

        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.carnum_webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.carnum_webcam_label.setPixmap(scaled_pixmap)

        # 计算 FPS
        self.carnum_webcam_frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.carnum_webcam_last_time
        fps = 0
        if elapsed >= 1.0:
            fps = self.carnum_webcam_frame_count / elapsed
            self.carnum_webcam_frame_count = 0
            self.carnum_webcam_last_time = current_time
            self.carnum_webcam_result.setText(f"未检测到目标\n检测速度: {fps:.1f} FPS")
            return

        detections = []
        if len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                conf_val = results[0].boxes.conf[i].cpu().numpy()
                cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                cls_name = model_carnum.names[cls_id]
                detections.append(f"{cls_name} ({conf_val:.2f})")
        
        result_text = "\n".join(detections) if detections else "未检测到目标"
        self.carnum_webcam_result.setText(result_text)

    # ===== 车牌检测 - 屏幕 =====
    def toggle_carnum_screen(self):
        global model_carnum

        if model_carnum is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        if self.is_carnum_screen_running:
            self.is_carnum_screen_running = False
            if self.carnum_screen_timer:
                self.carnum_screen_timer.stop()
            self.btn_carnum_screen.setText("开启屏幕检测")
            self.log("车牌检测屏幕检测已停止")
        else:
            # 重置 FPS 计数器
            self.carnum_screen_frame_count = 0
            self.carnum_screen_last_time = time.time()
            
            self.is_carnum_screen_running = True
            self.btn_carnum_screen.setText("停止屏幕检测")
            self.log("车牌检测屏幕检测已开启")

            self.carnum_screen_timer = QTimer()
            self.carnum_screen_timer.timeout.connect(self.update_carnum_screen)
            self.carnum_screen_timer.start(100)

    def update_carnum_screen(self):
        if not self.is_carnum_screen_running:
            return

        try:
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            conf = self.conf_slider.value() / 100
            results = model_carnum(frame, conf=conf)
            result_img = results[0].plot()

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.carnum_screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.carnum_screen_label.setPixmap(scaled_pixmap)

            # 计算 FPS
            self.carnum_screen_frame_count += 1
            current_time = time.time()
            elapsed = current_time - self.carnum_screen_last_time
            fps = 0
            if elapsed >= 1.0:
                fps = self.carnum_screen_frame_count / elapsed
                self.carnum_screen_frame_count = 0
                self.carnum_screen_last_time = current_time
                self.carnum_screen_result.setText(f"未检测到目标\n检测速度: {fps:.1f} FPS")
                return

            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model_carnum.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")
            
            result_text = "\n".join(detections) if detections else "未检测到目标"
            self.carnum_screen_result.setText(result_text)

        except Exception as e:
            self.log(f"车牌检测屏幕检测错误: {e}")

    def capture_carnum_screen(self):
        global model_carnum

        if model_carnum is None:
            QMessageBox.warning(self, "警告", "请先启动服务")
            return

        try:
            self.log("正在截取车牌屏幕...")
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            conf = self.conf_slider.value() / 100
            results = model_carnum(frame, conf=conf)
            result_img = results[0].plot()

            result_path = f"carnum_screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.jpg"
            cv2.imwrite(result_path, result_img)

            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.carnum_screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.carnum_screen_label.setPixmap(scaled_pixmap)

            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model_carnum.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")
            self.carnum_screen_result.setText("\n".join(detections) if detections else "未检测到目标")

            self.log(f"车牌屏幕截取完成: {result_path}")

        except Exception as e:
            self.log(f"车牌屏幕截取错误: {e}")
            QMessageBox.critical(self, "错误", f"屏幕截取失败: {e}")

    def closeEvent(self, event):
        if self.is_webcam_running and self.webcam_capture:
            self.webcam_capture.release()
        if self.webcam_timer:
            self.webcam_timer.stop()
        if self.screen_timer:
            self.screen_timer.stop()
        if self.is_carnum_webcam_running and self.carnum_webcam_capture:
            self.carnum_webcam_capture.release()
        if self.carnum_webcam_timer:
            self.carnum_webcam_timer.stop()
        if self.carnum_screen_timer:
            self.carnum_screen_timer.stop()
        event.accept()


if __name__ == '__main__':
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError:
        print("请安装 PyQt5: pip install PyQt5")
        sys.exit(1)

    try:
        from PIL import ImageGrab
    except ImportError:
        print("请安装 Pillow: pip install pillow")
        sys.exit(1)

    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
