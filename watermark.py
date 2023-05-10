import random
import cv2
import numpy as np
from numpy import ndarray

"""
DCT 水印嵌入方法2
将水印嵌入中频DCT系数
"""
np.random.seed(1)
x1 = np.random.randn(8)
x2 = np.random.randn(8)

# Zigzag矩阵
z_mat = np.array([
    0, 1, 5, 6, 14, 15, 27, 28,
    2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43,
    9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63])


# 相关系数
def corr2(a, b):
    a = a - np.mean(a)
    b = b - np.mean(b)
    r = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-20)
    return r


# zigzag 变换
def zig_zag(block: np.ndarray):
    zigzag_block = block.flatten()[[np.where(z_mat == i)[0][0] for i in range(64)]]
    return zigzag_block


# zigzag逆变换
def inverse_zigzag(arr: np.ndarray[np.uint8], shape=(8, 8)):
    rows, cols = shape
    out = np.zeros(shape)
    index = 0
    for i in range(rows + cols - 1):
        if i % 2 == 1:
            for j in range(min(i, rows - 1), max(0, i - cols + 1) - 1, -1):
                out[i - j][j] = arr[index]
                index += 1
        else:
            for j in range(max(0, i - cols + 1), min(i, rows - 1) + 1):
                out[i - j][j] = arr[index]
                index += 1
    return out


# 水印嵌入
def embed_watermark(image: np.ndarray[np.uint8],
                    watermark: np.ndarray[np.uint8],
                    alpha=0.5,
                    bs=8, seed=100) -> np.ndarray[np.uint8]:
    wm_copy = watermark.copy()
    img_f = image.astype(np.float32)  # 原始图像 float32
    # 图像尺寸
    img_h, img_w = image.shape
    # 水印尺寸
    wm_h, wm_w = watermark.shape

    blocks = (img_h // bs) * (img_w // bs)
    if watermark.size > blocks:
        exit("水印过大")
    img_idct = np.zeros_like(image, dtype=np.float32)  # 逆DCT变换后的图像
    for i in range(0, img_h, bs):
        for j in range(0, img_w, bs):
            dst = cv2.dct(img_f[i:i + bs, j:j + bs])
            img_idct[i:i + bs, j:j + bs] = cv2.idct(dst)

    # 水印二值化
    wm_copy[wm_copy < 127] = 0
    wm_copy[wm_copy >= 127] = 1

    used_point = set()
    random.seed(seed)  # 设置随机种子，确保嵌入与提取水印的位置相同
    n = img_h // bs
    m = img_w // bs
    p = (-1, -1)  # 初始点无意义
    used_point.add(p)
    for h in range(wm_h):
        for w in range(wm_w):
            # 如果该 8*8 DCT 块已经嵌入过水印则继续随机
            while p in used_point:
                p = random.randint(0, n - 1), random.randint(0, m - 1)
            used_point.add(p)

            index_h = p[0] * bs  # 水印像素将嵌入的DCT块的列索引
            index_w = p[1] * bs  # 水印像素将嵌入的DCT块的行索引

            dct_block = cv2.dct(img_idct[index_h:index_h + bs, index_w:index_w + bs])
            zigzag_block = zig_zag(dct_block)
            x = x1 if wm_copy[h, w] == 1 else x2
            zigzag_block[28:36] = zigzag_block[28:36] + (alpha * x)  # 主对角线中频系数
            inv_zigzag = inverse_zigzag(zigzag_block)
            img_idct[index_h:index_h + bs, index_w:index_w + bs] = cv2.idct(inv_zigzag)
    return np.uint8(img_idct)


# 水印提取
def extract_watermark(image: np.ndarray, wm_shape: tuple, bs=8, seed=100):
    wm_h, wm_w = wm_shape
    img_f = np.float32(image)
    random.seed(seed)
    used_point = set()
    p = (-1, -1)
    used_point.add(p)
    watermark = np.zeros(wm_shape, dtype=np.uint8)
    n = image.shape[0] // bs
    m = image.shape[1] // bs
    for h in range(wm_h):
        for w in range(wm_w):
            while p in used_point:
                p = random.randint(0, n - 1), random.randint(0, m - 1)
            used_point.add(p)
            index_h = int(p[0] * bs)
            index_w = int(p[1] * bs)

            dct_block = cv2.dct(img_f[index_h:index_h + bs, index_w:index_w + bs])
            zigzag_block = zig_zag(dct_block)
            val = zigzag_block[28:36]
            if corr2(val, x1) > corr2(val, x2):
                watermark[h, w] = 255
    return watermark


def embed_watermark_color(image: np.ndarray[np.uint8],
                          watermark: np.ndarray[np.uint8],
                          alpha=0.5,
                          bs=8, seed=100) -> ndarray[np.uint8]:
    """
    对彩色图像嵌入水印，同embed_watermark
    :param image: 图像
    :param watermark: 水印
    :param alpha: 强度
    :param bs: block size
    :param seed: 随机种子
    :return: 嵌入水印后图像
    """
    chs = [embed_watermark(image[:, :, i], watermark, alpha, bs, seed + i) for i in range(3)]
    return cv2.merge(chs)


def extract_watermark_color(image: np.ndarray, wm_shape: tuple, bs=8, seed=100):
    """
    从彩色图像中提取水印
    :param image: 图像
    :param wm_shape: 水印大小
    :param bs: block size
    :param seed: 随机种子
    :return: 水印
    """
    chs = [extract_watermark(image[:, :, i], wm_shape, bs, seed + i) for i in range(3)]
    a = np.logical_or(chs[0], chs[1])
    b = np.logical_or(chs[1], chs[2])
    c = np.logical_or(chs[0], chs[2])
    merged = np.logical_and(a, b, c).astype(np.uint8)
    return merged * 255, chs
