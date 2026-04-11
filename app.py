import os
import sys
import uuid
import jwt
import mimetypes
from datetime import datetime, timedelta,timezone
from functools import wraps
from pathlib import Path

import cv2
from flask import Flask, request, jsonify, send_file, current_app, url_for
from flask_cors import CORS
from ultralytics import YOLO

from database import (
    db,
    User, DetectionRecord,
    ROLE_USER, ROLE_ADMIN, ROLE_SUPER_ADMIN,
    hash_password,
    get_user_by_username, get_user_by_id,
    create_user, update_user_last_login,
    save_detection_record, get_history_list,
    get_record_by_id, delete_record_by_id,
)


def create_flask_app(model_type='yolo'):
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    CORS(app)
    app.config['MODEL_TYPE'] = model_type

    # 数据库配置 - 使用绝对路径
    basedir = os.path.abspath(os.path.dirname(__file__))
    db_path = os.path.join(basedir, 'database', 'db.sqlite3')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # JWT 配置
    app.config['JWT_SECRET_KEY'] = 'yolo_flask_jwt_secret_key_2026_yolo_api'  # 至少32字节
    app.config['JWT_EXPIRE_HOURS'] = 48               #token有效期
    db.init_app(app)

    with app.app_context():
        db.create_all()

    # 基础设置 - 使用绝对路径
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
    UPLOAD_FOLDER = os.path.join(basedir, 'static', 'uploads')
    RESULT_FOLDER = os.path.join(basedir, 'static', 'results')

    #创建文件夹
    for sub in ['images', 'videos']:
        os.makedirs(os.path.join(UPLOAD_FOLDER, sub), exist_ok=True)
        os.makedirs(os.path.join(RESULT_FOLDER, sub), exist_ok=True)

    #文件类型筛选
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

    #根据JWT创建token
    def generate_token(user_id, role):
        payload = {
            'user_id': user_id,
            'role':    role,
            'exp':     datetime.now(timezone.utc) + timedelta(hours=app.config['JWT_EXPIRE_HOURS']),
            'iat':     datetime.now(timezone.utc),
        }
        return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')
    #解码token
    def decode_token(token):
        return jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
    #token登录验证
    def login_required(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({'error': '未登录，请先登录'}), 401
            token = auth_header.split(' ', 1)[1]
            try:
                payload = decode_token(token)
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token 已过期，请重新登录'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Token 无效'}), 401
            user = get_user_by_id(payload['user_id'])
            if not user:
                return jsonify({'error': '账号不存在'}), 403
            request.current_user = user
            request.current_token = token
            return f(*args, **kwargs)
        return decorated

    #文件校验
    def allowed_file(filename, file_type):
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        if file_type == 'image':
            return ext in ALLOWED_IMAGE_EXTENSIONS
        elif file_type == 'video':
            return ext in ALLOWED_VIDEO_EXTENSIONS
        return False
    #文件保存
    def save_uploaded_file(file, file_type):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_id = uuid.uuid4().hex[:8]
        ext = file.filename.rsplit('.', 1)[1].lower()
        if file_type == 'image':
            filename = f'img_{timestamp}_{unique_id}.{ext}'
            filepath = os.path.join(UPLOAD_FOLDER, 'images', filename)
        else:
            filename = f'vid_{timestamp}_{unique_id}.{ext}'
            filepath = os.path.join(UPLOAD_FOLDER, 'videos', filename)
        file.save(filepath)
        return filepath, filename

    if model_type == 'carnum':
        model_file = 'yolo_carnum_best.pt'
    else:
        model_file = 'yolo26x.pt'
    model = YOLO(model_file)
    #图片检测方法
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
                detections.append({
                    'class':      model.names[cls_id],
                    'class_id':   cls_id,
                    'confidence': float(conf),
                    'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                })
        filename = Path(image_path).stem + '_annotated.jpg'
        result_path = os.path.join(RESULT_FOLDER, 'images', filename)
        cv2.imwrite(result_path, result_img)
        return detections, result_path, filename
    #视频检测方法
    def detect_video(video_path, conf_threshold=0.25):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        filename     = Path(video_path).stem + '_annotated.mp4'
        result_path  = os.path.join(RESULT_FOLDER, 'videos', filename)

        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
            if not out.isOpened():
                raise Exception("avc1 编码器无法打开")
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            filename    = filename.replace('.mp4', '.avi')
            result_path = result_path.replace('.mp4', '.avi')
            out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        all_detections  = []
        frame_count     = 0
        detect_interval = max(1, int(fps / 5))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, conf=conf_threshold)
            if frame_count % detect_interval == 0 and len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    x1, y1, x2, y2 = results[0].boxes.xyxy[i].cpu().numpy()
                    conf_val = results[0].boxes.conf[i].cpu().numpy()
                    cls_id   = int(results[0].boxes.cls[i].cpu().numpy())
                    all_detections.append({
                        'frame':      frame_count,
                        'class':      model.names[cls_id],
                        'class_id':   cls_id,
                        'confidence': float(conf_val),
                        'bbox': {'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2)}
                    })
            out.write(results[0].plot())
            frame_count += 1

        cap.release()
        out.release()
        return all_detections, result_path, filename


    #用户登录
    @app.route('/api/user/login', methods=['POST'])
    def login():
        data = request.get_json()
        if not data:
            return jsonify({'error': '请求体不能为空'}), 400
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        if not username or not password:
            return jsonify({'error': '用户名和密码不能为空'}), 400
        user = get_user_by_username(username)
        if not user or user.password != hash_password(password):
            return jsonify({'error': '用户名或密码错误'}), 401
        if not user.is_active:
            return jsonify({'error': '账号已被禁用'}), 403
        update_user_last_login(user)
        token = generate_token(user.id, user.role)
        return jsonify({'success': True, 'message': '登录成功', 'token': token, 'user': user.to_dict()})


    #图片检测api
    @app.route('/api/detect/image', methods=['POST'])
    @login_required
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
            try:
                # 保存检测记录时传入用户ID
                save_detection_record(
                    filename, result_filename,
                    current_app.config['MODEL_TYPE'], 'image',
                    user_id=request.current_user.id
                )
                print(f"✅ 图片记录已保存到数据库 (user_id={request.current_user.id})")
            except Exception as db_e:
                print("❌ 数据库保存失败：", str(db_e))
            return jsonify({
                'success':         True,
                'filename':        filename,
                'original_path':   filepath,
                'result_filename': result_filename,
                'detections':      detections,
                'count':           len(detections),
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    #视频检测api
    @app.route('/api/detect/video', methods=['POST'])
    @login_required
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
            try:
                # 保存检测记录时传入用户ID
                save_detection_record(
                    filename, result_filename,
                    current_app.config['MODEL_TYPE'], 'video',
                    user_id=request.current_user.id
                )
                print(f"✅ 视频记录已保存到数据库 (user_id={request.current_user.id})")
            except Exception as db_e:
                print("❌ 数据库保存失败：", str(db_e))
            return jsonify({
                'success':         True,
                'filename':        filename,
                'result_filename': result_filename,
                'result_url':      f'/api/result/{result_filename}',
                'original_path':   filepath,
                'detections':      detections,
                'count':           len(detections),
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': str(e)}), 500


    #图片查询api
    @app.route('/api/result/<filename>', methods=['GET'])
    def get_result(filename):
        if '/' in filename or '\\' in filename:
            return jsonify({'error': '非法的文件名'}), 400
        image_path = os.path.join(RESULT_FOLDER, 'images', filename)
        video_path = os.path.join(RESULT_FOLDER, 'videos', filename)
        target_path = image_path if os.path.exists(image_path) else (
            video_path if os.path.exists(video_path) else None
        )
        if not target_path:
            return jsonify({'error': '结果文件不存在'}), 404
        mime_type, _ = mimetypes.guess_type(target_path)
        if not mime_type:
            ext = filename.lower().rsplit('.', 1)[-1]
            mime_type = {'mp4': 'video/mp4', 'jpg': 'image/jpeg',
                         'jpeg': 'image/jpeg', 'png': 'image/png'}.get(ext, 'application/octet-stream')
        return send_file(target_path, mimetype=mime_type, as_attachment=False)
    #历史界面图片查询api
    @app.route('/api/resee/<int:id>', methods=['GET'])
    @login_required
    def resee(id):
        """查看检测结果，普通用户/管理员只能查看自己的记录"""
        record = get_record_by_id(id)
        if not record:
            return jsonify({'error': '当前记录不存在'}), 400

        # 权限检查：普通用户和管理员只能查看自己的记录
        user = request.current_user
        if user.role != ROLE_SUPER_ADMIN and record.user_id != user.id:
            return jsonify({'error': '无权限查看他人的检测记录'}), 403

        if record.detect_file_type == 'image':
            upload_path = os.path.join(UPLOAD_FOLDER, 'images', record.load_filename)
            result_path = os.path.join(RESULT_FOLDER, 'images', record.result_filename)
            if not os.path.exists(upload_path) or not os.path.exists(result_path):
                return jsonify({'error': '文件已丢失'}), 400
            return jsonify({
                'code':     200,
                'original': url_for('static', filename=f'uploads/images/{record.load_filename}'),
                'result':   url_for('static', filename=f'results/images/{record.result_filename}'),
            })
        elif record.detect_file_type == 'video':
            upload_path = os.path.join(UPLOAD_FOLDER, 'videos', record.load_filename)
            result_path = os.path.join(RESULT_FOLDER, 'videos', record.result_filename)
            if not os.path.exists(upload_path) or not os.path.exists(result_path):
                return jsonify({'error': '文件已丢失'}), 400
            return jsonify({
                'code':     200,
                'original': url_for('static', filename=f'uploads/videos/{record.load_filename}'),
                'result':   url_for('static', filename=f'results/videos/{record.result_filename}'),
            })
        return jsonify({'error': '不支持的文件类型'}), 400
    #历史记录获取api
    @app.route('/api/history/list', methods=['GET'])
    @login_required
    def get_history():
        """获取历史记录，普通用户/管理员只能看自己的，超级管理员可看所有"""
        user = request.current_user
        records = get_history_list(limit=100, user_id=user.id, role=user.role)
        return jsonify({
            'success': True,
            'list': [{
                'id':             r.id,
                'user_id':        r.user_id,
                'fileName':       r.load_filename,
                'resultFileName': r.result_filename,
                'modelType':      r.model_type,
                'detectFileType': r.detect_file_type,
                'createTime':     r.create_time.strftime('%Y-%m-%d %H:%M:%S'),
            } for r in records]
        })
    #历史数据删除api
    @app.route('/api/history/delete/<int:id>', methods=['GET'])
    @login_required
    def delete_history(id):
        """删除检测记录，普通用户/管理员只能删除自己的，超级管理员可删除所有"""
        record = get_record_by_id(id)
        if not record:
            return jsonify({'error': '当前记录不存在'}), 404

        # 权限检查：普通用户和管理员只能删除自己的记录
        user = request.current_user
        if user.role != ROLE_SUPER_ADMIN and record.user_id != user.id:
            return jsonify({'error': '无权限删除他人的检测记录'}), 403

        if record.detect_file_type == 'image':
            upload_path = os.path.join(UPLOAD_FOLDER, 'images', record.load_filename)
            result_path = os.path.join(RESULT_FOLDER, 'images', record.result_filename)
        elif record.detect_file_type == 'video':
            upload_path = os.path.join(UPLOAD_FOLDER, 'videos', record.load_filename)
            result_path = os.path.join(RESULT_FOLDER, 'videos', record.result_filename)
        else:
            return jsonify({'error': '不支持的文件类型'}), 400
        deleted_files = []
        for path, fname in [(upload_path, record.load_filename), (result_path, record.result_filename)]:
            if os.path.exists(path):
                os.remove(path)
                deleted_files.append(fname)
        try:
            delete_record_by_id(id)
            return jsonify({'success': True, 'message': f'记录 {id} 删除成功', 'deleted_files': deleted_files})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    #健康查询
    @app.route('/api/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status':    'ok',
            'model':     model_file,
            'timestamp': datetime.now().isoformat(),
        })

    return app


if __name__ == '__main__':
    port       = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    model_type = sys.argv[2]      if len(sys.argv) > 2 else 'yolo'
    flask_app  = create_flask_app(model_type)
    print(f"启动 {model_type} 模型服务，端口: {port}")
    flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
