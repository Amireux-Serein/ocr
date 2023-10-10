"""
@Project: ocr
@File   : enviroment_inspect.py
@Author : Ruiqing Tang
@Date   : 2023/10/4 9:46
"""
import os
import torch, torchvision
import mmcv
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
import mmocr
# 创建 checkpoint 文件夹，用于存放预训练模型权重文件
os.mkdir('checkpoint')

# 创建 outputs 文件夹，用于存放预测结果
os.mkdir('outputs')

# 创建 data 文件夹，用于存放图片和视频素材（未必会用到，我的data在移动硬盘里，记得改绝对路径）
os.mkdir('data')

print('Pytorch 版本', torch.__version__)
print('CUDA 是否可用',torch.cuda.is_available())

print('MMCV版本', mmcv.__version__)
print('CUDA版本', get_compiling_cuda_version())
print('编译器版本', get_compiler_version())

print('mmocr版本', mmocr.__version__)