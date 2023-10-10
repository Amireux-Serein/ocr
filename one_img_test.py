"""
@Project: ocr
@File   : one_img_test.py
@Author : Ruiqing Tang
@Date   : 2023/10/4 10:03
"""
import os
import cv2
import matplotlib.pyplot as plt
os.chdir('./mmocr')
# from mmocr.utils.ocr import MMOCR
from mmocr.apis import MMOCRInferencer
# 定义可视化图像函数，输入图像路径，可视化图像
def show_img_from_path(img_path):
    '''opencv 读入图像，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img = cv2.imread(img_path)
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

# 定义可视化图像函数，输入图像 array，可视化图像
def show_img_from_array(img):
    '''输入 array，matplotlib 可视化格式为 RGB，因此需将 BGR 转 RGB，最后可视化出来'''
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_RGB)
    plt.show()

show_img_from_path("D:\\Pycharm\\Projects\\ocr\\demo\\1.jpg")
detector = MMOCRInferencer(det='DBNet',  # 文本检测算法，这里指定为 TextSnake，也可替换为其他 MMOCR 支持的文本区域检测算法
                 rec='CRNN')       # 文本识别算法，这里不指定，也可替换为其他 MMOCR 支持的文本识别算法
                 # device='cuda')     # 指定运算设备为 cpu 或 cuda

