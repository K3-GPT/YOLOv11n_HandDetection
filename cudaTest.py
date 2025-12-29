import torch
print("PyTorch version:", torch.__version__)  # 输出 PyTorch 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
print(torch.cuda.device_count())  # 输出可用的 GPU 数量
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # 输出第一个 GPU 的名称