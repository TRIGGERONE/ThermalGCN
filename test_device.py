import torch
print("Torch version:", torch.__version__)  # PyTorch 版本
print("CUDA version:", torch.version.cuda)  # PyTorch 识别的 CUDA 版本
print("Is CUDA available:", torch.cuda.is_available())  # 是否可以使用 CUDA

