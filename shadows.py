from PIL import Image, ImageChops
import torchvision.transforms.functional as TF
import os
import numpy as np
import shutil
import argparse
import random
import string
import cv2
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser(description='add randomshadows processing')
    parser.add_argument('--images_path', type=str, default=r'G:/Deeplabv3plus/datasets/imgs')
    parser.add_argument('--labels_path', type=str, default=r'G:/Deeplabv3plus/datasets/labels')
    parser.add_argument('--output_save_root', type=str, default=r'G:/Deeplabv3plus/datasets/op')
    return parser

def contour(opt):
    '''
    主要功能:向原数据集中添加随机阴影。
    opt:参数
    '''
    palette = [0,0,0, 128,0,0, 0,128,0, 128,128,0, 0,0,128, 128,0,128, 0,128,128, 128,128,128] + 248 * [0, 0, 0]

    labels_path = opt.labels_path

    files = os.listdir(labels_path)
    for file in files:
        if file.endswith('.png'):
            filePrefix = file.split('.png')[0]
            label = np.asarray(Image.open(os.path.join(labels_path, filePrefix + '.png'))).astype(np.uint32)#提取label索引并以numpy形式保存
            mask = np.zeros((label.shape[0], label.shape[1]))#先置零，然后根据索引判断填充数组
            cls_list = np.unique(label)
            for idx in cls_list:
                if idx == 1:
                    mask[label == idx] = 1                   #提取人和动物的索引，其余清0
                # if idx == 2:
                #     mask[label == idx] = 2

            mask = Image.fromarray(mask)
            mask = mask.convert('P')
            mask.putpalette(palette)                         #提取出的目标轮廓转换为image彩图
            
            randomshadows(opt, mask, filePrefix)

def randomshadows(opt, mask, filePrefix):
    high_bright_factor = random.uniform(1, 1.5)
    low_bright_factor = random.uniform(0.2, 0.7)

    images_path = opt.images_path
    output_save_root = opt.output_save_root
    img1=Image.open('G:/Deeplabv3plus/datasets/imgs/000346.jpg')

    # 图像融合，将mask贴在原图上
    img0 = mask.convert("RGB")
    img2 = cv2.cvtColor(np.array(img0), cv2.COLOR_RGB2BGR)                  #image转cv2

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, tag = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)           #cv2.THRESH_OTSU, 二值化图像128  

    # 仿射变换
    height, width = tag.shape[:2]  # 405x413
    # 在原图像和目标图像上各选择三个点
    i = random.randint(0, 150)
    j = random.randint(0, 150)
    # print(i, j)
    matSrc = np.float32([[0, 0],[0, height-1],[width-1, 0]])
    matLUst = np.float32([[0, 0],[i, height-i],[width-j, j]])
    matRUst = np.float32([[i, height-i], [width-j, j], [0, 0]])

    matlist = [matLUst, matRUst]
    matOst = random.choice(matlist)
    # 得到变换矩阵
    matAffine = cv2.getAffineTransform(matSrc, matOst)
    # 进行仿射变换
    tag = cv2.warpAffine(tag, matAffine, (width,height))

    tag_inv = cv2.bitwise_not(tag)

    tag_pil = Image.fromarray(tag) 
    tag_inv_pil = Image.fromarray(tag_inv)         #查看二值化图像的转换

    tag_pil = tag_pil.convert("RGB")
    tag_inv_pil = tag_inv_pil.convert("RGB")
    
    #合成阴影
    low_brightness = TF.adjust_brightness(img1, low_bright_factor)
    low_brightness_masked = ImageChops.multiply(low_brightness, tag_pil)
    high_brightness = TF.adjust_brightness(img1, high_bright_factor)
    high_brightness_masked = ImageChops.multiply(high_brightness, tag_inv_pil)

    img_op = ImageChops.add(low_brightness_masked, high_brightness_masked)
    img_op.save(os.path.join(output_save_root, filePrefix + '.png'))

if __name__ == '__main__':
    parser = get_parser()
    opt = parser.parse_args()
    contour(opt)