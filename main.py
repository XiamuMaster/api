import sys
import os
import threading
import time
import cv2
import numpy as np
from datetime import datetime
import uuid

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QTabWidget, QMessageBox, QGroupBox,
                             QComboBox, QSlider, QProgressBar, QCheckBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

# YOLO 和 Flask 相关
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file, Flask
from flask_cors import CORS
import mimetypes
from pathlib import Path

# 屏幕截图
from PIL import ImageGrab, Image

# 全局变量
flask_thread = None
flask_app = None
model = None
is_detecting = False


# Flask 应用
def create_flask_app():
    app = Flask(__name__)
    CORS(app)

    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
    UPLOAD_FOLDER = 'uploads'
    RESULT_FOLDER = 'results'

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(RESULT_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(RESULT_FOLDER, 'videos'), exist_ok=True)

    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

    def allowed_file(filename, file_type):
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        if file_type == 'image':
            return ext in ALLOWED_IMAGE_EXTENSIONS
        elif file_type == 'video':
            return ext in ALLOWED_VIDEO_EXTENSIONS
        return False

    def save_uploaded_file(file, file_type):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]

        if file_type == 'image':
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f'img_{timestamp}_{unique_id}.{ext}'
            filepath = os.path.join(UPLOAD_FOLDER, 'images', filename)
        else:
            ext = file.filename.rsplit('.', 1)[1].lower()
            filename = f'vid_{timestamp}_{unique_id}.{ext}'
            filepath = os.path.join(UPLOAD_FOLDER, 'videos', filename)

        file.save(filepath)
        return filepath, filename

    def detect_image(image_path, conf_threshold=0.25):
        global model
        img = cv2.imread(image_path)
        results = model(img, conf=conf_threshold)

        detections = []
        result_img = results[0].plot()

        for result in results[0]:
            box = result.boxes
            if len(box) > 0:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = model.names[cls_id]
                detection = {
                    'class': cls_name,
                    'class_id': cls_id,
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(x1), 'y1': float(y1),
                        'x2': float(x2), 'y2': float(y2)
                    }
                }
                detections.append(detection)

        filename = Path(image_path).stem + '_annotated.jpg'
        result_path = os.path.join(RESULT_FOLDER, 'images', filename)
        result_filename = filename
        cv2.imwrite(result_path, result_img)

        return detections, result_path, result_filename

    def detect_video(video_path, conf_threshold=0.25):
        global model
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps == 0:
            fps = 30

        filename = Path(video_path).stem + '_annotated.mp4'
        result_path = os.path.join(RESULT_FOLDER, 'videos', filename)
        result_filename = filename

        # 使用 H.264 编码以兼容浏览器播放，添加备选编码
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1编码器无法打开")
        except Exception as e:
            print(f"avc1编码失败，尝试XVID: {e}")
            try:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter(result_path.replace('.mp4', '.avi'), fourcc, fps, (width, height))
                result_filename = result_filename.replace('.mp4', '.avi')
                result_path = result_path.replace('.mp4', '.avi')
            except:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        all_detections = []
        frame_count = 0
        detect_interval = max(1, int(fps / 5))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=conf_threshold)

            current_frame_detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model.names[cls_id]

                    detection = {
                        'frame': frame_count,
                        'class': cls_name,
                        'class_id': cls_id,
                        'confidence': float(conf_val),
                        'bbox': {
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2)
                        }
                    }
                    current_frame_detections.append(detection)

            if frame_count % detect_interval == 0:
                all_detections.extend(current_frame_detections)

            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            frame_count += 1

        cap.release()
        out.release()

        return all_detections, result_path, result_filename

    @app.route('/api/detect/image', methods=['POST'])
    def detect_image_api():
        global model
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        if not allowed_file(file.filename, 'image'):
            return jsonify({'error': '不支持的图片格式'}), 400

        filepath, filename = save_uploaded_file(file, 'image')

        try:
            conf = request.form.get('conf', 0.25, type=float)
            detections, result_path, result_filename = detect_image(filepath, conf)
            return jsonify({
                'success': True,
                'filename': filename,
                'original_path': filepath,
                'result_path': result_path,
                'result_filename': result_filename,
                'detections': detections,
                'count': len(detections)
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/detect/video', methods=['POST'])
    def detect_video_api():
        global model
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '文件名为空'}), 400
        if not allowed_file(file.filename, 'video'):
            return jsonify({'error': '不支持的视频格式'}), 400

        filepath, filename = save_uploaded_file(file, 'video')

        try:
            conf = request.form.get('conf', 0.25, type=float)
            detections, result_path, result_filename = detect_video(filepath, conf)

            # 构建视频访问URL
            result_url = f"/api/result/{result_filename}"

            return jsonify({
                'success': True,
                'filename': filename,
                'result_filename': result_filename,
                'result_url': result_url,
                'original_path': filepath,
                'result_path': result_path,
                'detections': detections,
                'count': len(detections)
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500

    @app.route('/api/result/<filename>', methods=['GET'])
    def get_result(filename):
        if '/' in filename or '\\' in filename:
            return jsonify({'error': '非法的文件名'}), 400

        image_path = os.path.join(RESULT_FOLDER, 'images', filename)
        video_path = os.path.join(RESULT_FOLDER, 'videos', filename)

        target_path = None

        if os.path.exists(image_path):
            target_path = image_path
        elif os.path.exists(video_path):
            target_path = video_path

        if not target_path:
            return jsonify({'error': '结果文件不存在'}), 404

        mime_type, _ = mimetypes.guess_type(target_path)

        if not mime_type:
            if filename.lower().endswith('.mp4'):
                mime_type = 'video/mp4'
            elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                mime_type = 'image/jpeg'
            elif filename.lower().endswith('.png'):
                mime_type = 'image/png'
            else:
                mime_type = 'application/octet-stream'

        return send_file(target_path, mimetype=mime_type, as_attachment=False)

    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'ok',
            'model': 'yolo26m.pt',
            'timestamp': datetime.now().isoformat()
        })

    return app


# 检测线程
class DetectionThread(QThread):
    """后台检测线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, list)  # image_path, detections
    error = pyqtSignal(str)

    def __init__(self, mode, file_path=None, conf=0.25, webcam_frame=None):
        super().__init__()
        self.mode = mode  # 'image', 'video', 'webcam', 'screen'
        self.file_path = file_path
        self.conf = conf
        self.webcam_frame = webcam_frame

    def run(self):
        global model
        try:
            if self.mode == 'image':
                self.progress.emit("正在检测图片...")
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

                # 保存结果
                result_path = self.file_path.replace('.', '_annotated.')
                cv2.imwrite(result_path, result_img)

                self.finished.emit(result_path, detections)

            elif self.mode == 'video':
                self.progress.emit("正在检测视频，请稍候...")
                cap = cv2.VideoCapture(self.file_path)

                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                if fps == 0:
                    fps = 30

                # 正确处理文件扩展名
                base, ext = os.path.splitext(self.file_path)
                result_path = f"{base}_annotated{ext}"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

                all_detections = []
                frame_count = 0
                detect_interval = max(1, int(fps / 5))

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
                        self.progress.emit(f"已处理 {frame_count} 帧...")

                cap.release()
                out.release()

                self.finished.emit(result_path, all_detections)

            elif self.mode == 'webcam':
                self.progress.emit("正在检测摄像头画面...")
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
                self.progress.emit("正在检测屏幕...")
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

                # 转换结果图像为 Qt 格式
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                h, w, ch = result_rgb.shape
                qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)

                self.finished.emit("", detections)

        except Exception as e:
            self.error.emit(str(e))


# 主窗口
class YOLOApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.detection_thread = None
        self.webcam_timer = None
        self.screen_timer = None
        self.webcam_capture = None
        self.is_webcam_running = False
        self.is_screen_running = False

    def initUI(self):
        self.setWindowTitle('YOLO26 智能物品识别系统')
        self.setGeometry(100, 100, 1200, 800)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)

        # ===== 控制面板 =====
        control_group = QGroupBox("控制面板")
        control_layout = QHBoxLayout()

        # 启动 Flask 服务按钮
        self.btn_start_flask = QPushButton("启动 Flask 服务")
        self.btn_start_flask.clicked.connect(self.start_flask)
        control_layout.addWidget(self.btn_start_flask)

        # 置信度滑块
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

        # ===== 标签页 =====
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # ===== 文件检测标签页 =====
        self.tab_file = QWidget()
        self.tabs.addTab(self.tab_file, "文件检测")
        self.init_file_tab()

        # ===== 摄像头检测标签页 =====
        self.tab_webcam = QWidget()
        self.tabs.addTab(self.tab_webcam, "摄像头检测")
        self.init_webcam_tab()

        # ===== 屏幕检测标签页 =====
        self.tab_screen = QWidget()
        self.tabs.addTab(self.tab_screen, "屏幕检测")
        self.init_screen_tab()

        # ===== 日志标签页 =====
        self.tab_log = QWidget()
        self.tabs.addTab(self.tab_log, "运行日志")
        self.init_log_tab()

        # ===== 状态栏 =====
        self.statusBar().showMessage("就绪")

    def init_file_tab(self):
        layout = QVBoxLayout()

        # 按钮区域
        btn_layout = QHBoxLayout()

        self.btn_select_image = QPushButton("选择图片检测")
        self.btn_select_image.clicked.connect(lambda: self.select_file('image'))
        btn_layout.addWidget(self.btn_select_image)

        self.btn_select_video = QPushButton("选择视频检测")
        self.btn_select_video.clicked.connect(lambda: self.select_file('video'))
        btn_layout.addWidget(self.btn_select_video)

        layout.addLayout(btn_layout)

        # 结果显示区域 - 使用StackedLayout切换图片和视频
        self.result_stacked_widget = QWidget()
        self.result_stack_layout = QVBoxLayout()
        self.result_stacked_widget.setLayout(self.result_stack_layout)

        # 图片显示
        self.file_result_label = QLabel("请选择图片或视频进行检测")
        self.file_result_label.setAlignment(Qt.AlignCenter)
        self.file_result_label.setMinimumHeight(400)
        self.file_result_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        self.result_stack_layout.addWidget(self.file_result_label)

        # 视频播放器
        from PyQt5.QtMultimediaWidgets import QVideoWidget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(400)
        self.video_widget.setStyleSheet("border: 1px solid #ccc; background: #000;")
        self.video_widget.setVisible(False)
        self.result_stack_layout.addWidget(self.video_widget)

        # 视频播放控制
        self.video_player = QMediaPlayer()
        self.video_player.setVideoOutput(self.video_widget)
        self.video_player.stateChanged.connect(self.on_video_state_changed)

        layout.addWidget(self.result_stacked_widget)

        # 检测结果文本
        self.file_result_text = QTextEdit()
        self.file_result_text.setMaximumHeight(150)
        self.file_result_text.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.file_result_text)

        # 进度条
        self.file_progress = QProgressBar()
        self.file_progress.setVisible(False)
        layout.addWidget(self.file_progress)

        self.tab_file.setLayout(layout)

    def init_webcam_tab(self):
        layout = QVBoxLayout()

        # 按钮区域
        btn_layout = QHBoxLayout()

        self.btn_webcam = QPushButton("开启摄像头检测")
        self.btn_webcam.clicked.connect(self.toggle_webcam)
        btn_layout.addWidget(self.btn_webcam)

        layout.addLayout(btn_layout)

        # 摄像头画面
        self.webcam_label = QLabel("点击按钮开启摄像头")
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumHeight(400)
        self.webcam_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.webcam_label)

        # 检测结果
        self.webcam_result_text = QTextEdit()
        self.webcam_result_text.setMaximumHeight(100)
        self.webcam_result_text.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.webcam_result_text)

        self.tab_webcam.setLayout(layout)

    def init_screen_tab(self):
        layout = QVBoxLayout()

        # 按钮区域
        btn_layout = QHBoxLayout()

        self.btn_screen = QPushButton("开始屏幕检测")
        self.btn_screen.clicked.connect(self.toggle_screen)
        btn_layout.addWidget(self.btn_screen)

        self.btn_capture = QPushButton("截取当前屏幕")
        self.btn_capture.clicked.connect(self.capture_screen)
        btn_layout.addWidget(self.btn_capture)

        layout.addLayout(btn_layout)

        # 屏幕画面
        self.screen_label = QLabel("点击按钮开始屏幕检测")
        self.screen_label.setAlignment(Qt.AlignCenter)
        self.screen_label.setMinimumHeight(400)
        self.screen_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        layout.addWidget(self.screen_label)

        # 检测结果
        self.screen_result_text = QTextEdit()
        self.screen_result_text.setMaximumHeight(100)
        self.screen_result_text.setReadOnly(True)
        layout.addWidget(QLabel("检测结果:"))
        layout.addWidget(self.screen_result_text)

        self.tab_screen.setLayout(layout)

    def init_log_tab(self):
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 清空日志按钮
        btn_clear = QPushButton("清空日志")
        btn_clear.clicked.connect(lambda: self.log_text.clear())
        layout.addWidget(btn_clear)

        self.tab_log.setLayout(layout)

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def start_flask(self):
        global flask_thread

        if flask_thread and flask_thread.is_already_running:
            self.log("Flask 服务已在运行中")
            return

        # 启动 Flask 线程
        flask_thread = FlaskThread()
        flask_thread.log_signal.connect(self.log)
        flask_thread.error_signal.connect(lambda e: self.on_flask_error(e))
        flask_thread.start()

        self.btn_start_flask.setEnabled(False)
        self.btn_start_flask.setText("Flask 服务运行中")

    def on_flask_error(self, error):
        self.btn_start_flask.setEnabled(True)
        self.btn_start_flask.setText("启动 Flask 服务")
        QMessageBox.critical(self, "错误", f"启动失败: {error}")

    def select_file(self, file_type):
        global model

        if model is None:
            QMessageBox.warning(self, "警告", "请先启动 Flask 服务")
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
        self.file_progress.setVisible(True)
        self.file_progress.setRange(0, 0)  # 不确定进度

        # 启动检测线程
        self.detection_thread = DetectionThread(file_type, file_path, conf)
        self.detection_thread.progress.connect(lambda m: self.log(m))
        self.detection_thread.finished.connect(lambda path, dets: self.on_detection_finished(path, dets))
        self.detection_thread.error.connect(lambda e: self.on_detection_error(e))
        self.detection_thread.start()

    def on_detection_finished(self, result_path, detections):
        self.file_progress.setVisible(False)
        self.log(f"检测完成! 发现 {len(detections)} 个目标")

        # 显示检测结果
        result_text = f"检测到 {len(detections)} 个目标:\n"
        for i, det in enumerate(detections[:10]):  # 最多显示10个
            result_text += f"{i + 1}. {det['class']} - {det['confidence']:.2f}\n"

        if len(detections) > 10:
            result_text += f"... 还有 {len(detections) - 10} 个目标"

        self.file_result_text.setText(result_text)

        # 显示结果 - 根据文件类型判断是图片还是视频
        if result_path and os.path.exists(result_path):
            # 检查文件扩展名
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
            is_video = any(result_path.lower().endswith(ext) for ext in video_extensions)

            if is_video:
                # 视频播放
                self.log(f"正在播放视频: {result_path}")
                self.file_result_label.setVisible(False)
                self.video_widget.setVisible(True)
                self.video_player.setMedia(QMediaContent(QUrl.fromLocalFile(result_path)))
                self.video_player.play()
            else:
                # 图片显示
                self.video_player.stop()
                self.video_widget.setVisible(False)
                self.file_result_label.setVisible(True)
                pixmap = QPixmap(result_path)
                if not pixmap.isNull():
                    # 缩放图片适应显示
                    scaled_pixmap = pixmap.scaled(
                        self.file_result_label.size(),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    self.file_result_label.setPixmap(scaled_pixmap)

    def on_video_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            self.log("视频播放完成")

    def on_detection_error(self, error):
        self.file_progress.setVisible(False)
        self.log(f"检测错误: {error}")
        QMessageBox.critical(self, "错误", f"检测失败: {error}")

    def toggle_webcam(self):
        global model

        if model is None:
            QMessageBox.warning(self, "警告", "请先启动 Flask 服务")
            return

        if self.is_webcam_running:
            # 停止摄像头
            self.is_webcam_running = False
            if self.webcam_capture:
                self.webcam_capture.release()
            if self.webcam_timer:
                self.webcam_timer.stop()
            self.btn_webcam.setText("开启摄像头检测")
            self.webcam_label.setText("摄像头已关闭")
            self.log("摄像头检测已停止")
        else:
            # 开启摄像头
            self.webcam_capture = cv2.VideoCapture(0)
            if not self.webcam_capture.isOpened():
                QMessageBox.critical(self, "错误", "无法打开摄像头")
                return

            self.is_webcam_running = True
            self.btn_webcam.setText("关闭摄像头检测")
            self.log("摄像头检测已开启")

            # 定时器更新画面
            self.webcam_timer = QTimer()
            self.webcam_timer.timeout.connect(self.update_webcam)
            self.webcam_timer.start(30)  # 约30fps

    def update_webcam(self):
        if not self.is_webcam_running or not self.webcam_capture:
            return

        ret, frame = self.webcam_capture.read()
        if not ret:
            return

        # 复制帧用于检测
        frame_copy = frame.copy()

        # 检测
        conf = self.conf_slider.value() / 100
        results = model(frame_copy, conf=conf)
        result_img = results[0].plot()

        # 显示结果
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        h, w, ch = result_rgb.shape
        qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        scaled_pixmap = pixmap.scaled(
            self.webcam_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.webcam_label.setPixmap(scaled_pixmap)

        # 显示检测结果
        detections = []
        if len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                conf_val = results[0].boxes.conf[i].cpu().numpy()
                cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                cls_name = model.names[cls_id]
                detections.append(f"{cls_name} ({conf_val:.2f})")

        self.webcam_result_text.setText("\n".join(detections) if detections else "未检测到目标")

    def toggle_screen(self):
        global model

        if model is None:
            QMessageBox.warning(self, "警告", "请先启动 Flask 服务")
            return

        if self.is_screen_running:
            self.is_screen_running = False
            if self.screen_timer:
                self.screen_timer.stop()
            self.btn_screen.setText("开始屏幕检测")
            self.screen_label.setText("屏幕检测已停止")
            self.log("屏幕检测已停止")
        else:
            self.is_screen_running = True
            self.btn_screen.setText("停止屏幕检测")
            self.log("屏幕检测已开启")

            self.screen_timer = QTimer()
            self.screen_timer.timeout.connect(self.update_screen)
            self.screen_timer.start(100)  # 约10fps

    def update_screen(self):
        if not self.is_screen_running:
            return

        try:
            # 截取屏幕
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 检测
            conf = self.conf_slider.value() / 100
            results = model(frame, conf=conf)
            result_img = results[0].plot()

            # 显示结果
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.screen_label.setPixmap(scaled_pixmap)

            # 显示检测结果
            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")

            self.screen_result_text.setText("\n".join(detections) if detections else "未检测到目标")

        except Exception as e:
            self.log(f"屏幕检测错误: {e}")

    def capture_screen(self):
        """截取当前屏幕并进行检测"""
        global model

        if model is None:
            QMessageBox.warning(self, "警告", "请先启动 Flask 服务")
            return

        try:
            self.log("正在截取屏幕...")
            screen = ImageGrab.grab()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            conf = self.conf_slider.value() / 100
            results = model(frame, conf=conf)
            result_img = results[0].plot()

            # 保存结果
            result_path = f"screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_annotated.jpg"
            cv2.imwrite(result_path, result_img)

            # 显示结果
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            h, w, ch = result_rgb.shape
            qimg = QImage(result_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            scaled_pixmap = pixmap.scaled(
                self.screen_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.screen_label.setPixmap(scaled_pixmap)

            # 显示检测结果
            detections = []
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id = int(results[0].boxes.cls[i].cpu().numpy())
                    cls_name = model.names[cls_id]
                    detections.append(f"{cls_name} ({conf_val:.2f})")

            self.screen_result_text.setText("\n".join(detections) if detections else "未检测到目标")

            self.log(f"屏幕截取完成，结果已保存: {result_path}")

        except Exception as e:
            self.log(f"屏幕截取错误: {e}")
            QMessageBox.critical(self, "错误", f"屏幕截取失败: {e}")

    def closeEvent(self, event):
        # 关闭时清理资源
        if self.is_webcam_running and self.webcam_capture:
            self.webcam_capture.release()
        if self.webcam_timer:
            self.webcam_timer.stop()
        if self.screen_timer:
            self.screen_timer.stop()
        event.accept()


# Flask 线程
class FlaskThread(QThread):
    log_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    is_already_running = False

    def __init__(self):
        super().__init__()

    def run(self):
        global flask_app, model
        try:
            self.log_signal.emit("正在加载 YOLO 模型...")

            # 加载模型
            if model is None:
                model = YOLO('yolo26m.pt')

            self.log_signal.emit("YOLO 模型加载成功!")
            self.log_signal.emit("正在启动 Flask 服务...")

            flask_app = create_flask_app()
            FlaskThread.is_already_running = True
            self.log_signal.emit("Flask 服务已启动: http://127.0.0.1:5000")

            # 启动 Flask
            flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

        except Exception as e:
            error_msg = str(e)
            self.log_signal.emit(f"启动失败: {error_msg}")
            self.error_signal.emit(error_msg)
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    # 检查依赖
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
