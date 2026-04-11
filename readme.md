# YOLO 目标检测系统

基于 PyQt5 + Flask + YOLO 的目标检测系统，支持普通目标检测和车牌检测两种模式。

## 功能特性

### 检测模式
- **普通检测**：通用目标检测（端口 5000）
- **车牌检测**：车牌识别专用模型（端口 5001）

### 支持的检测方式
- 图片上传检测
- 视频上传检测
- 屏幕实时检测（PyQt5 界面）
- 摄像头实时检测

### 用户权限系统
- **超级管理员 (super_admin)**：查看所有用户的检测记录，删除任意记录
- **管理员 (admin)**：仅查看和删除自己的检测记录
- **普通用户 (user)**：仅查看和删除自己的检测记录

### 其他功能
- JWT Token 认证（48小时有效期）
- 检测历史记录管理
- 原图与检测结果对比查看

## 目录结构

```
yoloapi/
├── main.py                  # PyQt5 主界面程序
├── app.py                   # Flask API 后端
├── database/                # 数据库包
│   ├── __init__.py
│   ├── models.py            # 数据模型
│   ├── crud.py              # 数据库操作
│   └── db.sqlite3           # SQLite 数据库文件
├── static/                  # 静态文件
│   └── uploads/             # 上传文件
│   └── results/             # 检测结果
├── yolo26x.pt               # 普通目标检测模型
├── yolo_carnum_best.pt      # 车牌检测模型
└── requirements.txt         # 依赖包
```

## 安装步骤

### 1. 创建虚拟环境（推荐）

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 准备模型文件

将 YOLO 模型文件放在项目根目录：
- `yolo26x.pt` - 普通目标检测模型
- `yolo_carnum_best.pt` - 车牌检测模型

## 运行方式

### 启动主程序

```bash
python main.py
```

这将启动 PyQt5 桌面界面，包含：
- 模式切换（普通检测 / 车牌检测）
- 文件上传检测
- 屏幕实时检测
- 历史记录管理
- 用户管理（超级管理员）

### 单独启动 Flask API

```bash
# 普通检测模式（端口 5000）
python app.py yolo

# 车牌检测模式（端口 5001）
python app.py carnum
```

## API 接口

> 所有需要认证的接口需要在 Header 中携带 Token：
> ```
> Authorization: Bearer <token>
> ```

### 认证接口

#### 登录
```
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

#### 登出
```
POST /api/auth/logout
Authorization: Bearer <token>
```

### 检测接口

#### 图片检测
```
POST /api/detect/image
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: 图片文件
conf: 置信度阈值（可选，默认 0.25）
```

#### 视频检测
```
POST /api/detect/video
Authorization: Bearer <token>
Content-Type: multipart/form-data

file: 视频文件
conf: 置信度阈值（可选，默认 0.25）
```

### 历史记录接口

#### 查询历史
```
GET /api/history/list?limit=100
Authorization: Bearer <token>
```

响应示例：
```json
{
  "records": [
    {
      "id": 1,
      "load_filename": "img_xxx.jpg",
      "result_filename": "result_xxx.jpg",
      "detect_file_type": "image",
      "create_time": "2026-04-11T12:00:00",
      "user_id": 1
    }
  ]
}
```

#### 删除记录
```
GET /api/history/delete/<id>
Authorization: Bearer <token>
```

#### 查看原图/结果
```
GET /api/resee/<id>
Authorization: Bearer <token>
```

### 其他接口

#### 健康检查
```
GET /api/health
```

## 默认账号

| 用户名 | 密码 | 角色 |
|--------|------|------|
| admin | admin123 | super_admin |
| user1 | user123 | user |
| admin2 | admin123 | admin |

> 首次启动服务时自动创建超级管理员账号

## 文件格式支持

### 图片格式
- JPG / JPEG
- PNG
- BMP
- WebP

### 视频格式
- MP4
- AVI
- MOV
- MKV
- FLV

## 配置说明

### 数据库
- 路径：`database/db.sqlite3`
- 类型：SQLite

### 文件存储
- 上传文件：`static/uploads/images/` 和 `static/uploads/videos/`
- 检测结果：`static/results/images/` 和 `static/results/videos/`

### 文件大小限制
- 最大支持 500MB

## 技术栈

- **后端**：Flask + Flask-CORS + Flask-SQLAlchemy
- **前端界面**：PyQt5
- **目标检测**：Ultralytics YOLO
- **认证**：PyJWT
- **数据库**：SQLite + SQLAlchemy

## 注意事项

1. **GPU 加速**：如果有 NVIDIA GPU，系统会自动使用 CUDA 加速
2. **内存占用**：处理大视频时需要足够的内存
3. **Token 刷新**：Token 过期后需要重新登录
4. **删除操作**：删除记录会同时删除原始文件和结果文件
