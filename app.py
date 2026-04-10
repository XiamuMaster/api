import os
import uuid
from datetime import datetime
from pathlib import Path
import cv2
import mimetypes
from flask import Flask, request, jsonify, send_file, current_app
from flask_cors import CORS
from ultralytics import YOLO
from flask_sqlalchemy import SQLAlchemy

#初始化数据库
db = SQLAlchemy()

class DetectionRecord(db.Model):
    __tablename__ = 'YOLOlist'
    id = db.Column(db.Integer, primary_key=True)
    load_filename = db.Column(db.String(255), nullable=False)  # 原文件名
    result_filename = db.Column(db.String(255))                # 结果文件名
    model_type = db.Column(db.String(255))                     # 模型类型
    detect_file_type = db.Column(db.String(255))               # 文件类型
    create_time = db.Column(db.DateTime, default=datetime.now)

# Flask 应用
def create_flask_app(model_type='yolo'):
    app = Flask(__name__)
    CORS(app)
    app.config['MODEL_TYPE'] = model_type

    #数据库配置
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)

    with app.app_context():
        db.create_all()
        print('✅ 数据库表创建完成')

    #flask基础设置
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

    # 根据模型类型选择模型文件
    if model_type == 'carnum':
        model_file = 'yolo_carnum_best.pt'
    else:
        model_file = 'yolo26x.pt'

    model = YOLO(model_file)

    # 检测文件后缀
    def allowed_file(filename, file_type):
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        if file_type == 'image':
            return ext in ALLOWED_IMAGE_EXTENSIONS
        elif file_type == 'video':
            return ext in ALLOWED_VIDEO_EXTENSIONS
        return False

    #保存上传
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

    # 图片检测
    def detect_image(image_path, conf_threshold=0.25):
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

            # ======================
            # 数据库写入（已修复）
            # ======================
            try:
                current_model = current_app.config['MODEL_TYPE']
                record = DetectionRecord(
                    load_filename=filename,
                    result_filename=result_filename,
                    model_type=current_model,
                    detect_file_type='image'
                )
                db.session.add(record)
                db.session.commit()
                print("✅ 图片记录已保存到数据库")
            except Exception as db_e:
                print("❌ 数据库保存失败：", str(db_e))
                db.session.rollback()

            return jsonify({
                'success': True,
                'filename': filename,
                'original_path': filepath,
                'result_filename': result_filename,
                'detections': detections,
                'count': len(detections),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/detect/video', methods=['POST'])
    def detect_video_api():
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

            # ======================
            # 数据库写入（已修复）
            # ======================
            try:
                current_model = current_app.config['MODEL_TYPE']
                record = DetectionRecord(
                    load_filename=filename,
                    result_filename=result_filename,
                    model_type=current_model,
                    detect_file_type='video'  # 这里修复了！
                )
                db.session.add(record)
                db.session.commit()
                print("✅ 视频记录已保存到数据库")
            except Exception as db_e:
                print("❌ 数据库保存失败：", str(db_e))
                db.session.rollback()

            result_url = f"/api/result/{result_filename}"

            return jsonify({
                'success': True,
                'filename': filename,
                'result_filename': result_filename,
                'result_url': result_url,
                'original_path': filepath,
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

    @app.route('/api/resee/<filename>', methods=['GET'])
    def resee(filename):
        if not filename:
            return jsonify({"error": "请传入 filename 参数"}), 400
        image_path = os.path.join(UPLOAD_FOLDER, 'images', filename)
        video_path = os.path.join(UPLOAD_FOLDER, 'videos', filename)
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
    @app.route('/api/history/list', methods=['GET'])
    def get_history():
        records = DetectionRecord.query.order_by(DetectionRecord.create_time.desc()).limit(20).all()
        result = []
        for record in records:
            result.append({
                "id": record.id,
                "fileName": record.load_filename,  # 原文件名
                "resultFileName": record.result_filename,  # 结果文件名
                "modelType": record.model_type,  # yolo / carnum
                "detectFileType": record.detect_file_type,  # image / video
                "createTime": record.create_time.strftime("%Y-%m-%d %H:%M:%S")
            })

        return jsonify({
            "success": True,
            "list": result
        })
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'ok',
            'model': model_file,
            'timestamp': datetime.now().isoformat()
        })

    return app


# 根据不同端口启动不同模型的服务
if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'yolo'

    app = create_flask_app(model_type)
    print(f"启动 {model_type} 模型服务，端口: {port}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)