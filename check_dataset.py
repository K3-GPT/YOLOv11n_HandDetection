#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查手部检测数据集的结构和内容
"""

import zipfile
import os
import json
import shutil

def check_dataset():
    dataset_path = "d:/Python_Files/Personal_projects/YOLOv8/hand_detection_dataset"
    
    print("=== 手部检测数据集检查报告 ===\n")
    
    # 检查zip文件
    zip_path = os.path.join(dataset_path, "to_coco.zip")
    if os.path.exists(zip_path):
        print(f"✅ 找到压缩文件: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                files = z.namelist()
                print(f"📦 Zip文件包含 {len(files)} 个文件")
                print("\n前20个文件:")
                for f in files[:20]:
                    print(f"  {f}")
                if len(files) > 20:
                    print(f"  ... 还有 {len(files)-20} 个文件")
        except Exception as e:
            print(f"❌ 读取zip文件出错: {e}")
    else:
        print(f"❌ 压缩文件不存在: {zip_path}")
    
    # 检查现有的val2017目录
    val_path = os.path.join(dataset_path, "val2017")
    if os.path.exists(val_path):
        val_images = [f for f in os.listdir(val_path) if f.endswith('.jpg')]
        print(f"\n✅ 找到验证集目录: {val_path}")
        print(f"📸 验证集包含 {len(val_images)} 张图片")
    else:
        print(f"\n❌ 验证集目录不存在: {val_path}")
    
    # 检查配置文件
    config_path = os.path.join(dataset_path, "hand_detection_dataset.json")
    if os.path.exists(config_path):
        print(f"\n✅ 找到配置文件: {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                print("📋 数据集配置信息:")
                print(f"  - 训练集: {config['default'].get('train', {})}")
                print(f"  - 验证集: {config['default'].get('validation', {})}")
        except Exception as e:
            print(f"❌ 读取配置文件出错: {e}")
    else:
        print(f"\n❌ 配置文件不存在: {config_path}")

if __name__ == "__main__":
    check_dataset()