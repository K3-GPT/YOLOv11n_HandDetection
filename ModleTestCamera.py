import cv2
import time
from ultralytics import YOLO

def realtime_hand_detection(model_path="yolo11n_hand_detect.pt", conf_threshold=0.4):
    """
    实时手部检测程序（笔记本摄像头）
    按空格键退出
    """
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("模型加载成功！")

    # 打开摄像头（0通常是内置摄像头）
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        return

    print("\n摄像头已启动")
    print("=" * 50)
    print("实时检测中... (按空格键退出)")
    print("=" * 50)

    # 用于计算FPS
    fps_start_time = time.time()
    fps_counter = 0

    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧，结束程序")
                break

            # 进行检测
            results = model(frame, conf=conf_threshold, verbose=False)

            # 获取检测结果
            detections = results[0].boxes
            num_hands = len(detections)

            # 绘制检测框和信息
            annotated_frame = results[0].plot()  # 自动绘制框和标签

            # 计算FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            else:
                fps = 0

            # 在左上角显示信息
            cv2.putText(annotated_frame, f"Hands: {num_hands}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Press SPACE to exit",
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示画面
            cv2.imshow('YOLOv8 Hand Detection - Press SPACE to Exit', annotated_frame)

            # 检查按键 - 空格键退出 (ASCII 32)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # 空格键
                print("\n检测到空格键，程序结束")
                break
            elif key == ord('q'):  # 也可以按'q'退出
                print("\n检测到'q'键，程序结束")
                break

    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        print("\n摄像头已关闭，程序退出")

if __name__ == "__main__":
    # 配置参数
    # MODEL_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\runs\detect\yolo11n_hand_detect.pt2\weights\last.pt" # 模型路径
    MODEL_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\runs\detect\yolo11n_hand_detect.pt2\weights\best.pt" # 模型路径
    CONFIDENCE = 0.4  # 置信度阈值（0.2-0.5之间调整）

    # 启动实时检测
    realtime_hand_detection(MODEL_PATH, CONFIDENCE)
