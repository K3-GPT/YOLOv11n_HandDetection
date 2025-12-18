import cv2
from ultralytics import YOLO

def detect_hands_and_show(image_path, model_path="yolo11n_hand_detect.pt", conf_threshold=0.4):
    """
    读取本地图片，检测手部，在左上角显示信息，并以500x500窗口显示
    """
    # 加载模型
    model = YOLO(model_path)

    # 读取图片
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误：无法读取图片 - {image_path}")
        return

    # 进行检测
    results = model(frame, conf=conf_threshold, verbose=False)

    # 绘制检测框
    annotated_frame = results[0].plot()

    # 获取检测结果
    detections = results[0].boxes
    num_hands = len(detections)

    # ==================== 在左上角绘制信息 ====================
    # 绘制半透明黑色背景
    overlay = annotated_frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 70 + num_hands * 25), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

    # 显示检测数量
    cv2.putText(annotated_frame, f"Hands: {num_hands}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 显示每个手部的置信度
    for i, box in enumerate(detections):
        conf = box.conf[0].item()
        cv2.putText(annotated_frame, f"Conf: {conf:.2f}",
                    (15, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    if num_hands == 0:
        cv2.putText(annotated_frame, "No hands detected",
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # ==================== 调整窗口大小为500x500 ====================
    annotated_frame = cv2.resize(annotated_frame, (500, 500))

    # 显示结果
    cv2.imshow('Hand Detection Result - Press any key to close', annotated_frame)
    print(f"检测完成！共检测到 {num_hands} 个手部")
    print("窗口大小: 500x500")
    print("按任意键关闭窗口...")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def batch_detect(image_folder, model_path="yolo11n_hand_detect.pt", conf_threshold=0.4):
    """
    批量检测文件夹中的所有图片（500x500窗口，逐张显示）
    """
    import os
    from pathlib import Path

    model = YOLO(model_path)

    # 获取所有图片
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [f for f in Path(image_folder).iterdir()
                   if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"错误：在 {image_folder} 中未找到图片")
        return

    print(f"\n找到 {len(image_files)} 张图片，开始检测...")

    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] 处理: {image_path.name}")

        # 读取图片
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("  跳过：无法读取")
            continue

        # 检测
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = results[0].plot()

        # 获取信息
        detections = results[0].boxes
        num_hands = len(detections)

        # 在左上角绘制信息
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (5, 5), (280, 70 + num_hands * 25), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

        cv2.putText(annotated_frame, f"Hands: {num_hands}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        for i, box in enumerate(detections):
            conf = box.conf[0].item()
            cv2.putText(annotated_frame, f"Conf: {conf:.2f}",
                        (15, 60 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if num_hands == 0:
            cv2.putText(annotated_frame, "No hands detected",
                        (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # 调整大小为500x500
        annotated_frame = cv2.resize(annotated_frame, (500, 500))

        # 显示
        cv2.imshow(f'Result {idx}/{len(image_files)} - Press any key or q to skip/quit', annotated_frame)

        # 等待用户操作
        key = cv2.waitKey(0)
        if key == ord('q'):  # 按q退出
            print("用户按q键，停止批量处理")
            break

        cv2.destroyAllWindows()


def main():
    # ==================== 配置区域 ====================

    # 模式选择: "single" (单张图片) 或 "batch" (批量处理文件夹)
    MODE = "single"

    # 模型路径
    MODEL_PATH =  r"D:\Python_Files\Personal_projects\YOLOv8\runs\detect\yolo11n_hand_detect.pt2\weights\last.pt"

    # 置信度阈值 (0.2-0.5之间调整)
    CONF_THRESHOLD = 0.4

    # 单张图片模式
    if MODE == "single":
        # IMAGE_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\plam.jpg"  # 修改为你的图片文件夹路径
        IMAGE_PATH = r"D:\Python_Files\Personal_projects\YOLOv8\1.jpg"  # 修改为你的图片路径
        # IMAGE_PATH = "D:/images/my_photo.jpg"  # 或使用绝对路径

        detect_hands_and_show(
            image_path=IMAGE_PATH,
            model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD
        )

    # 批量处理模式
    elif MODE == "batch":
        IMAGE_FOLDER = "test_images"  # 修改为你的图片文件夹路径
        # IMAGE_FOLDER = "D:/images/hand_dataset"  # 或使用绝对路径

        batch_detect(
            image_folder=IMAGE_FOLDER,
            model_path=MODEL_PATH,
            conf_threshold=CONF_THRESHOLD
        )

    else:
        print("错误：MODE 必须是 'single' 或 'batch'")


if __name__ == "__main__":
    main()
