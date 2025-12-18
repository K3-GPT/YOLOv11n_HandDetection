'''
from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS  # 添加跨域支持
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os
import traceback

# ==================== 安装依赖 ====================
# 如果尚未安装，请运行以下命令：
# pip install flask flask-cors opencv-python ultralytics numpy
# =================================================

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 加载YOLO模型
MODEL_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\runs\detect\yolo11n_hand_detect.pt2\weights\last.pt"
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 未找到模型文件 {MODEL_PATH}")
    print("请先下载模型或修改MODEL_PATH为正确的路径")
    exit(1)

model = YOLO(MODEL_PATH)
print(f"✅ 模型加载成功: {MODEL_PATH}")

# HTML界面模板（增强版）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>手部检测 - 摄像头模式</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-size: 14px;
        }
        .status-panel {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
            font-size: 14px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        #startBtn {
            background: #4CAF50;
            color: white;
        }
        #startBtn:hover:not(:disabled) {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        #startBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #stopBtn {
            background: #f44336;
            color: white;
        }
        #stopBtn:hover:not(:disabled) {
            background: #da190b;
            transform: translateY(-2px);
        }
        #stopBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .display-area {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .display-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .display-box h3 {
            margin: 0 0 10px 0;
            color: #555;
            font-size: 16px;
        }
        video, canvas {
            border: 2px solid #ddd;
            border-radius: 8px;
            background: #000;
            max-width: 100%;
            height: auto;
        }
        #status {
            font-weight: bold;
            color: #333;
        }
        .status-active { color: #4CAF50 !important; }
        .status-inactive { color: #f44336 !important; }
        .status-warning { color: #ff9800 !important; }

        #info {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            font-size: 15px;
            line-height: 1.6;
            min-height: 60px;
            white-space: pre-line;
            font-family: 'Courier New', monospace;
        }
        .info-empty { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .info-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info-debug { background: #e7f3ff; color: #0066cc; border: 1px solid #bee5eb; }

        .instructions {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            color: #0066cc;
        }
        .instructions ol {
            margin: 10px 0;
            padding-left: 20px;
        }
        .instructions li {
            margin: 5px 0;
        }
        .debug-log {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 手部实时检测系统</h1>
        <div class="subtitle">基于YOLO的Web摄像头检测</div>

        <div class="instructions">
            <strong>使用说明：</strong>
            <ol>
                <li>点击"打开摄像头"按钮授权浏览器访问摄像头</li>
                <li>等待视频流稳定后会自动开始检测</li>
                <li>点击"停止检测"可关闭摄像头</li>
                <li>检测信息会实时显示在下方</li>
                <li>如果失败，请查看浏览器控制台(F12)和下方调试日志</li>
            </ol>
        </div>

        <div class="controls">
            <button id="startBtn" onclick="startCamera()">🎥 打开摄像头并开始检测</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>⏹️ 停止检测</button>
        </div>

        <div class="status-panel">
            <div><strong>当前状态：</strong> <span id="status">等待启动...</span></div>
            <div style="margin-top: 5px;"><strong>已处理帧数：</strong> <span id="frameCount">0</span></div>
        </div>

        <div class="display-area">
            <div class="display-box">
                <h3>摄像头画面 (原始)</h3>
                <video id="video" width="400" height="300" autoplay playsinline muted></video>
            </div>

            <div class="display-box">
                <h3>检测结果 (500x500)</h3>
                <canvas id="resultCanvas" width="500" height="500"></canvas>
                <div id="info" class="info-empty">等待检测...</div>
            </div>
        </div>

        <div class="debug-log" id="debugLog">=== 调试日志将显示在这里 ===</div>
    </div>

    <script>
        let video = document.getElementById('video');
        let resultCanvas = document.getElementById('resultCanvas');
        let ctx = resultCanvas.getContext('2d');
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        let frameCount = 0;
        let videoReady = false;

        // 调试日志函数
        function logDebug(message) {
            const debugLog = document.getElementById('debugLog');
            const timestamp = new Date().toLocaleTimeString();
            debugLog.textContent += '[' + timestamp + '] ' + message + '\\n';
            debugLog.scrollTop = debugLog.scrollHeight;
            console.log('[DEBUG] ' + message);
        }

        async function startCamera() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const info = document.getElementById('info');

            logDebug('=== 开始启动摄像头 ===');
            startBtn.disabled = true;
            status.textContent = '正在请求摄像头权限...';
            status.className = 'status-warning';

            try {
                logDebug('请求getUserMedia...');
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'  // 优先使用前置摄像头
                    },
                    audio: false
                });

                logDebug('摄像头权限已获取，设置视频源');
                video.srcObject = stream;

                // 等待视频加载完成
                await new Promise((resolve, reject) => {
                    video.onloadedmetadata = () => {
                        videoReady = true;
                        logDebug('视频元数据加载完成: ' + video.videoWidth + 'x' + video.videoHeight);
                        resolve();
                    };
                    video.onerror = (e) => {
                        reject(new Error('视频加载失败'));
                    };
                    // 超时保护
                    setTimeout(() => {
                        if (!videoReady) reject(new Error('视频加载超时'));
                    }, 5000);
                });

                await video.play();
                logDebug('视频开始播放');

                // 更新UI
                stopBtn.disabled = false;
                status.textContent = '摄像头已启动，检测中...';
                status.className = 'status-active';
                info.textContent = '正在检测手部...';
                info.className = 'info-success';

                isRunning = true;
                frameCount = 0;
                document.getElementById('frameCount').textContent = '0';

                // 开始处理循环
                logDebug('启动处理循环，间隔150ms');
                intervalId = setInterval(captureAndProcess, 150);

            } catch (err) {
                logDebug('❌ 错误: ' + err.message);
                console.error('摄像头错误:', err);
                startBtn.disabled = false;
                status.textContent = '错误: ' + err.message;
                status.className = 'status-inactive';
                info.textContent = '启动失败: ' + err.message + '\\n请检查摄像头权限和连接。';
                info.className = 'info-error';
            }
        }

        function stopCamera() {
            logDebug('=== 停止检测 ===');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const info = document.getElementById('info');

            isRunning = false;
            videoReady = false;

            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
                logDebug('处理循环已停止');
            }

            if (stream) {
                stream.getTracks().forEach(track => {
                    track.stop();
                    logDebug('轨道已停止: ' + track.kind);
                });
                video.srcObject = null;
                stream = null;
            }

            startBtn.disabled = false;
            stopBtn.disabled = true;
            status.textContent = '检测已停止';
            status.className = 'status-inactive';
            info.textContent = '摄像头已关闭';
            info.className = 'info-empty';
        }

        async function captureAndProcess() {
            if (!isRunning || !videoReady) {
                if (!videoReady) logDebug('警告: 视频未就绪，跳过帧');
                return;
            }

            try {
                // 绘制当前视频帧到canvas
                ctx.drawImage(video, 0, 0, resultCanvas.width, resultCanvas.height);

                // 转换为Blob
                resultCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        logDebug('❌ Canvas转Blob失败');
                        return;
                    }

                    logDebug('📦 准备发送帧数据: ' + blob.size + ' bytes');

                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');

                    try {
                        logDebug('🚀 发送POST请求到 /process...');
                        const response = await fetch('/process', {
                            method: 'POST',
                            body: formData
                        });

                        if (!response.ok) {
                            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
                        }

                        const data = await response.json();
                        logDebug('✅ 响应接收成功: ' + JSON.stringify(data));

                        if (data.success) {
                            frameCount++;
                            document.getElementById('frameCount').textContent = frameCount;

                            const infoDiv = document.getElementById('info');

                            if (data.num_hands > 0) {
                                let infoText = '✅ 检测到 ' + data.num_hands + ' 个手部\\n';
                                data.confidences.forEach((conf, i) => {
                                    infoText += '   手部' + (i+1) + ': ' + conf.toFixed(2) + '\\n';
                                });
                                infoDiv.textContent = infoText;
                                infoDiv.className = 'info-success';
                            } else {
                                infoDiv.textContent = '❌ 未检测到手部';
                                infoDiv.className = 'info-empty';
                            }
                        } else {
                            throw new Error(data.error || '处理失败');
                        }

                    } catch (error) {
                        logDebug('❌ POST请求失败: ' + error.message);
                        console.error('请求错误:', error);
                        const infoDiv = document.getElementById('info');
                        infoDiv.textContent = '网络错误: ' + error.message + '\\n检查后端是否运行正常。';
                        infoDiv.className = 'info-error';
                    }
                }, 'image/jpeg', 0.8);

            } catch (err) {
                logDebug('❌ 捕获帧失败: ' + err.message);
                console.error('捕获错误:', err);
            }
        }

        // 页面关闭时自动停止
        window.addEventListener('beforeunload', () => {
            if (isRunning) stopCamera();
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """返回HTML界面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/process', methods=['POST', 'OPTIONS'])
def process_frame():
    """处理前端发送的图像帧"""
    # 处理预检请求（CORS）
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        print("\n" + "="*50)
        print("📥 收到新的图像处理请求")

        # 检查是否有文件
        if 'image' not in request.files:
            print("❌ 错误: 请求中没有image文件")
            return jsonify({'success': False, 'error': 'No image data received'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        print(f"✅ 接收到图像数据: {len(image_bytes)} bytes")

        # 转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ 错误: 无法解码图像")
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        print(f"✅ 图像解码成功: {frame.shape}")

        # 使用YOLO检测
        print("🧠 开始YOLO推理...")
        results = model(frame, conf=0.4, verbose=False)

        # 获取检测结果
        detections = results[0].boxes
        num_hands = len(detections)
        confidences = [box.conf[0].item() for box in detections]

        print(f"✅ 检测完成: {num_hands} 个手部, 置信度 {confidences}")

        # 返回JSON结果
        response = jsonify({
            'success': True,
            'num_hands': num_hands,
            'confidences': confidences
        })

        print("="*50)
        return response

    except Exception as e:
        print(f"❌ 处理异常: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def main():
    print("=" * 70)
    print("🚀 手部检测Web服务启动中...")
    print("=" * 70)
    print("\n✅ 服务已就绪！请在浏览器中访问:")
    print("   🔗 http://127.0.0.1:5000")
    print("\n📱 也可在同一局域网的其他设备访问:")
    print("   🔗 http://192.168.46.108:5000")
    print("\n💡 使用提示:")
    print("   1. 点击'打开摄像头'按钮")
    print("   2. 授权浏览器访问摄像头")
    print("   3. 开始实时手部检测！")
    print("   4. 如果失败，查看浏览器F12控制台和下方调试日志")
    print("\n按 Ctrl+C 可停止服务")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    main()
'''

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS  # 添加跨域支持
import cv2
import numpy as np
from ultralytics import YOLO
import io
import os
import traceback

# ==================== 安装依赖 ====================
# 如果尚未安装，请运行以下命令：
# pip install flask flask-cors opencv-python ultralytics numpy
# =================================================

app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 加载YOLO模型
MODEL_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\runs\detect\yolo11n_hand_detect.pt2\weights\best.pt"
if not os.path.exists(MODEL_PATH):
    print(f"❌ 错误: 未找到模型文件 {MODEL_PATH}")
    print("请先下载模型或修改MODEL_PATH为正确的路径")
    exit(1)

model = YOLO(MODEL_PATH)
print(f"✅ 模型加载成功: {MODEL_PATH}")

# HTML界面模板（增强版，包含绘图功能）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>手部检测 - 摄像头模式</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            max-width: 900px;
            width: 100%;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 25px;
            font-size: 14px;
        }
        .status-panel {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
            border-left: 4px solid #667eea;
            font-size: 14px;
        }
        .controls {
            display: flex;
            gap: 10px;
            justify-content: center;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        button {
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        #startBtn {
            background: #4CAF50;
            color: white;
        }
        #startBtn:hover:not(:disabled) {
            background: #45a049;
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        }
        #startBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        #stopBtn {
            background: #f44336;
            color: white;
        }
        #stopBtn:hover:not(:disabled) {
            background: #da190b;
            transform: translateY(-2px);
        }
        #stopBtn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .display-area {
            display: flex;
            gap: 20px;
            margin-top: 20px;
            flex-wrap: wrap;
            justify-content: center;
        }
        .display-box {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .display-box h3 {
            margin: 0 0 10px 0;
            color: #555;
            font-size: 16px;
        }
        video, canvas {
            border: 2px solid #ddd;
            border-radius: 8px;
            background: #000;
            max-width: 100%;
            height: auto;
        }
        #status {
            font-weight: bold;
            color: #333;
        }
        .status-active { color: #4CAF50 !important; }
        .status-inactive { color: #f44336 !important; }
        .status-warning { color: #ff9800 !important; }

        #info {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            font-size: 15px;
            line-height: 1.6;
            min-height: 60px;
            white-space: pre-line;
            font-family: 'Courier New', monospace;
        }
        .info-empty { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
        .info-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .info-error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info-debug { background: #e7f3ff; color: #0066cc; border: 1px solid #bee5eb; }

        .instructions {
            background: #e7f3ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 14px;
            color: #0066cc;
        }
        .instructions ol {
            margin: 10px 0;
            padding-left: 20px;
        }
        .instructions li {
            margin: 5px 0;
        }
        .debug-log {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📸 手部实时检测系统</h1>
        <div class="subtitle">基于YOLO的Web摄像头检测</div>

        <div class="instructions">
            <strong>使用说明：</strong>
            <ol>
                <li>点击"打开摄像头"按钮授权浏览器访问摄像头</li>
                <li>等待视频流稳定后会自动开始检测</li>
                <li>点击"停止检测"可关闭摄像头</li>
                <li>检测信息会实时显示在下方，手部边界框会直接绘制在右侧画布上</li>
                <li>如果失败，请查看浏览器控制台(F12)和下方调试日志</li>
            </ol>
        </div>

        <div class="controls">
            <button id="startBtn" onclick="startCamera()">🎥 打开摄像头并开始检测</button>
            <button id="stopBtn" onclick="stopCamera()" disabled>⏹️ 停止检测</button>
        </div>

        <div class="status-panel">
            <div><strong>当前状态：</strong> <span id="status">等待启动...</span></div>
            <div style="margin-top: 5px;"><strong>已处理帧数：</strong> <span id="frameCount">0</span></div>
        </div>

        <div class="display-area">
            <div class="display-box">
                <h3>摄像头画面 (原始)</h3>
                <video id="video" width="400" height="300" autoplay playsinline muted></video>
            </div>

            <div class="display-box">
                <h3>检测结果 (500x500)</h3>
                <!-- 注意：我们将在canvas上直接绘制检测结果 -->
                <canvas id="resultCanvas" width="500" height="500"></canvas>
                <div id="info" class="info-empty">等待检测...</div>
            </div>
        </div>

        <div class="debug-log" id="debugLog">=== 调试日志将显示在这里 ===</div>
    </div>

    <script>
        let video = document.getElementById('video');
        let resultCanvas = document.getElementById('resultCanvas');
        let ctx = resultCanvas.getContext('2d');
        let stream = null;
        let isRunning = false;
        let intervalId = null;
        let frameCount = 0;
        let videoReady = false;

        // 调试日志函数
        function logDebug(message) {
            const debugLog = document.getElementById('debugLog');
            const timestamp = new Date().toLocaleTimeString();
            debugLog.textContent += '[' + timestamp + '] ' + message + '\\n';
            debugLog.scrollTop = debugLog.scrollHeight;
            console.log('[DEBUG] ' + message);
        }

        async function startCamera() {
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const info = document.getElementById('info');

            logDebug('=== 开始启动摄像头 ===');
            startBtn.disabled = true;
            status.textContent = '正在请求摄像头权限...';
            status.className = 'status-warning';

            try {
                logDebug('请求getUserMedia...');
                stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'  // 优先使用前置摄像头
                    },
                    audio: false
                });

                logDebug('摄像头权限已获取，设置视频源');
                video.srcObject = stream;

                // 等待视频加载完成
                await new Promise((resolve, reject) => {
                    video.onloadedmetadata = () => {
                        videoReady = true;
                        logDebug('视频元数据加载完成: ' + video.videoWidth + 'x' + video.videoHeight);
                        resolve();
                    };
                    video.onerror = (e) => {
                        reject(new Error('视频加载失败'));
                    };
                    // 超时保护
                    setTimeout(() => {
                        if (!videoReady) reject(new Error('视频加载超时'));
                    }, 5000);
                });

                await video.play();
                logDebug('视频开始播放');

                // 更新UI
                stopBtn.disabled = false;
                status.textContent = '摄像头已启动，检测中...';
                status.className = 'status-active';
                info.textContent = '正在检测手部...';
                info.className = 'info-success';

                isRunning = true;
                frameCount = 0;
                document.getElementById('frameCount').textContent = '0';

                // 开始处理循环
                logDebug('启动处理循环，间隔150ms');
                intervalId = setInterval(captureAndProcess, 150);

            } catch (err) {
                logDebug('❌ 错误: ' + err.message);
                console.error('摄像头错误:', err);
                startBtn.disabled = false;
                status.textContent = '错误: ' + err.message;
                status.className = 'status-inactive';
                info.textContent = '启动失败: ' + err.message + '\\n请检查摄像头权限和连接。';
                info.className = 'info-error';
            }
        }

        function stopCamera() {
            logDebug('=== 停止检测 ===');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const status = document.getElementById('status');
            const info = document.getElementById('info');

            isRunning = false;
            videoReady = false;

            if (intervalId) {
                clearInterval(intervalId);
                intervalId = null;
                logDebug('处理循环已停止');
            }

            if (stream) {
                stream.getTracks().forEach(track => {
                    track.stop();
                    logDebug('轨道已停止: ' + track.kind);
                });
                video.srcObject = null;
                stream = null;
            }

            startBtn.disabled = false;
            stopBtn.disabled = true;
            status.textContent = '检测已停止';
            status.className = 'status-inactive';
            info.textContent = '摄像头已关闭';
            info.className = 'info-empty';
        }

        async function captureAndProcess() {
            if (!isRunning || !videoReady) {
                if (!videoReady) logDebug('警告: 视频未就绪，跳过帧');
                return;
            }

            try {
                // 绘制当前视频帧到canvas (这是原始帧)
                ctx.drawImage(video, 0, 0, resultCanvas.width, resultCanvas.height);

                // 转换为Blob
                resultCanvas.toBlob(async (blob) => {
                    if (!blob) {
                        logDebug('❌ Canvas转Blob失败');
                        return;
                    }

                    logDebug('📦 准备发送帧数据: ' + blob.size + ' bytes');

                    const formData = new FormData();
                    formData.append('image', blob, 'frame.jpg');

                    try {
                        logDebug('🚀 发送POST请求到 /process...');
                        const response = await fetch('/process', {
                            method: 'POST',
                            body: formData
                        });

                        if (!response.ok) {
                            throw new Error('HTTP ' + response.status + ': ' + response.statusText);
                        }

                        const data = await response.json();
                        logDebug('✅ 响应接收成功: ' + JSON.stringify(data));

                        if (data.success) {
                            frameCount++;
                            document.getElementById('frameCount').textContent = frameCount;

                            // 清空画布以准备绘制新结果 (再次绘制原始帧，因为toBlob会清空画布)
                            ctx.drawImage(video, 0, 0, resultCanvas.width, resultCanvas.height);

                            // 绘制检测框
                            if (data.boxes && data.boxes.length > 0) {
                                ctx.strokeStyle = '#FF0000'; // 红色
                                ctx.lineWidth = 2;

                                // 为了在不同尺寸的画布上正确缩放坐标，我们需要计算缩放比例
                                // 假设原图是640x480 (摄像头的常见分辨率)，画布是500x500
                                const srcW = video.videoWidth || 640; // 如果无法获取，则假设为640
                                const srcH = video.videoHeight || 480; // 如果无法获取，则假设为480
                                const dstW = resultCanvas.width; // 500
                                const dstH = resultCanvas.height; // 500

                                // 计算缩放比例 (保持宽高比，居中放置)
                                const scale = Math.min(dstW / srcW, dstH / srcH);
                                const offsetX = (dstW - srcW * scale) / 2;
                                const offsetY = (dstH - srcH * scale) / 2;

                                for (let i = 0; i < data.boxes.length; i++) {
                                    const box = data.boxes[i];
                                    // 原始坐标是 [x1, y1, x2, y2]
                                    // 计算缩放后的坐标
                                    const x1 = offsetX + box[0] * scale;
                                    const y1 = offsetY + box[1] * scale;
                                    const x2 = offsetX + box[2] * scale;
                                    const y2 = offsetY + box[3] * scale;

                                    // 绘制矩形框
                                    ctx.beginPath();
                                    ctx.rect(x1, y1, x2 - x1, y2 - y1);
                                    ctx.stroke();

                                    // 在框上方绘制置信度标签
                                    ctx.fillStyle = 'rgba(255, 0, 0, 0.75)';
                                    ctx.font = '12px Arial';
                                    const label = 'Hand ' + (i+1) + ': ' + data.confidences[i].toFixed(2);
                                    const labelMetrics = ctx.measureText(label);
                                    ctx.fillRect(x1, y1 - 14, labelMetrics.width + 4, 14); // 背景矩形

                                    ctx.fillStyle = 'white';
                                    ctx.fillText(label, x1 + 2, y1 - 4); // 文本
                                }

                                // 更新信息面板
                                let infoText = '✅ 检测到 ' + data.num_hands + ' 个手部\\n';
                                data.confidences.forEach((conf, i) => {
                                    infoText += '   手部' + (i+1) + ': ' + conf.toFixed(2) + '\\n';
                                });
                                document.getElementById('info').textContent = infoText;
                                document.getElementById('info').className = 'info-success';
                            } else {
                                document.getElementById('info').textContent = '❌ 未检测到手部';
                                document.getElementById('info').className = 'info-empty';
                            }
                        } else {
                            throw new Error(data.error || '处理失败');
                        }

                    } catch (error) {
                        logDebug('❌ POST请求失败: ' + error.message);
                        console.error('请求错误:', error);
                        const infoDiv = document.getElementById('info');
                        infoDiv.textContent = '网络错误: ' + error.message + '\\n检查后端是否运行正常。';
                        infoDiv.className = 'info-error';
                    }
                }, 'image/jpeg', 0.8);

            } catch (err) {
                logDebug('❌ 捕获帧失败: ' + err.message);
                console.error('捕获错误:', err);
            }
        }

        // 页面关闭时自动停止
        window.addEventListener('beforeunload', () => {
            if (isRunning) stopCamera();
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """返回HTML界面"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST', 'OPTIONS'])
def process_frame():
    """处理前端发送的图像帧"""
    # 处理预检请求（CORS）
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        print("\n" + "=" * 50)
        print("📥 收到新的图像处理请求")

        # 检查是否有文件
        if 'image' not in request.files:
            print("❌ 错误: 请求中没有image文件")
            return jsonify({'success': False, 'error': 'No image data received'}), 400

        image_file = request.files['image']
        image_bytes = image_file.read()

        print(f"✅ 接收到图像数据: {len(image_bytes)} bytes")

        # 转换为numpy数组
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            print("❌ 错误: 无法解码图像")
            return jsonify({'success': False, 'error': 'Failed to decode image'}), 400

        print(f"✅ 图像解码成功: {frame.shape}")

        # 使用YOLO检测
        print("🧠 开始YOLO推理...")
        results = model(frame, conf=0.4, verbose=False)

        # 获取检测结果
        detections = results[0].boxes
        num_hands = len(detections)

        boxes = []
        confidences = []
        if detections is not None and len(detections) > 0:
            # 获取边界框坐标 (xyxy格式: [x1, y1, x2, y2])
            boxes_data = detections.xyxy.cpu().numpy()
            confs_data = detections.conf.cpu().numpy()

            for i in range(len(boxes_data)):
                box = boxes_data[i].tolist()  # [x1, y1, x2, y2]
                conf = confs_data[i].item()

                # 确保坐标在图像范围内
                h, w = frame.shape[:2]
                x1 = max(0, min(box[0], w))
                y1 = max(0, min(box[1], h))
                x2 = max(0, min(box[2], w))
                y2 = max(0, min(box[3], h))

                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                confidences.append(float(conf))

        print(f"✅ 检测完成: {num_hands} 个手部, 框坐标: {boxes}, 置信度: {confidences}")

        # 返回JSON结果，包含边界框坐标
        response_data = {
            'success': True,
            'num_hands': num_hands,
            'confidences': confidences,
            'boxes': boxes  # 添加边界框坐标
        }

        response = jsonify(response_data)
        print("=" * 50)
        return response

    except Exception as e:
        print(f"❌ 处理异常: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    print("=" * 70)
    print("🚀 手部检测Web服务启动中...")
    print("=" * 70)
    print("\n✅ 服务已就绪！请在浏览器中访问:")
    print("   🔗 http://127.0.0.1:5000")
    print("\n📱 也可在同一局域网的其他设备访问:")
    print("   🔗 http://192.168.46.108:5000")
    print("\n💡 使用提示:")
    print("   1. 点击'打开摄像头'按钮")
    print("   2. 授权浏览器访问摄像头")
    print("   3. 开始实时手部检测！检测到的手部会被红色框标记出来")
    print("   4. 如果失败，查看浏览器F12控制台和下方调试日志")
    print("\n按 Ctrl+C 可停止服务")
    print("=" * 70)

    app.run(host='0.0.0.0', port=5000, debug=False)


if __name__ == '__main__':
    main()