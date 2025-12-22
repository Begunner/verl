import torch
import os

# 必须先触发 cudnn 加载
print(f"CuDNN Version: {torch.backends.cudnn.version()}")
# 随便移一个 tensor 到 cuda 来确保库被初始化
try:
    torch.zeros(1).cuda()
except:
    pass 

print("\n--- 实际加载的库文件路径 ---")
# 查看当前进程加载的内存映射，过滤出 cudnn
os.system(f"cat /proc/{os.getpid()}/maps | grep libcudnn")
