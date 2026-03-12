YOLO 检测 API 后端
功能特性
支持图片上传和检测
支持视频上传和检测
自动保存原始文件和标注后的文件
返回详细的检测结果（类别、置信度、边界框坐标）
目录结构
Claw/
├── app.py              # 主程序
├── requirements.txt    # 依赖包
├── yolov6n.pt         # YOLO 模型文件
├── uploads/            # 上传文件存储
│   ├── images/
│   └── videos/
└── results/           # 检测结果存储
    ├── images/
    └── videos/
安装步骤
1. 安装依赖
bash
复制
pip install -r requirements.txt
2. 准备模型
将训练好的 YOLO 模型放在项目根目录，命名为 yolov6n.pt

或者使用官方模型：

bash
复制
# 首次运行会自动下载
python app.py
运行服务
bash
复制
python app.py
服务启动后运行在：http://127.0.0.1:5000

API 接口
1. 健康检查
GET /api/health
响应：

json
复制
{
  "status": "ok",
  "model": "yolov6n.pt",
  "timestamp": "2026-03-10T10:30:00"
}
2. 图片检测
POST /api/detect/image
参数：

file: 图片文件（multipart/form-data）
conf: 置信度阈值（可选，默认 0.25）
请求示例：

bash
复制
curl -X POST http://127.0.0.1:5000/api/detect/image \
  -F "file=@test.jpg" \
  -F "conf=0.5"
响应示例：

json
复制
3. 视频检测
POST /api/detect/video
参数：

file: 视频文件（multipart/form-data）
conf: 置信度阈值（可选，默认 0.25）
请求示例：

bash
复制
curl -X POST http://127.0.0.1:5000/api/detect/video \
  -F "file=@test.mp4" \
  -F "conf=0.5"
响应示例：

json
复制
4. 获取结果文件
GET /api/result/<filename>
请求示例：

bash
复制
# 获取标注后的图片
curl http://127.0.0.1:5000/api/result/img_20260310_103000_a1b2c3d4_annotated.jpg

# 获取标注后的视频
curl http://127.0.0.1:5000/api/result/vid_20260310_103000_a1b2c3d4_annotated.mp4 -o output.mp4
支持的格式
图片格式
JPG / JPEG
PNG
BMP
WebP
视频格式
MP4
AVI
MOV
MKV
FLV
前端集成示例
JavaScript / Fetch
javascript
复制
Python / Requests
python
复制
注意事项
文件大小限制：最大支持 500MB
视频处理时间：视频检测可能需要较长时间，请耐心等待
GPU 加速：如果有 NVIDIA GPU，会自动使用 CUDA 加速
内存占用：处理大视频时需要足够的内存
故障排除
模型加载失败
确保 yolov6n.pt 文件在项目根目录

CORS 错误
API 已配置 CORS，前端可直接跨域访问

视频处理失败
检查视频格式和编码是否被 OpenCV 支持