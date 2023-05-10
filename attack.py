import numpy as np


def poisson_noisy(img):
    """
    对图像加入泊松噪声
    :param img: 图像
    :return: 加入噪声后图像
    """
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_img = np.random.poisson(img * vals) / float(vals)
    return noisy_img.astype(np.uint8)


def gauss_noisy(img, mean=0, sigma=0.01):
    """
    对图像加入高斯噪声
    :param img: 图像
    :param mean: 高斯噪声均值
    :param sigma: 高斯噪声标准差
    :return: 加入噪声后图像
    """
    noisy = np.random.normal(mean, sigma, img.shape)
    noisy_img = img + noisy
    noisy_img = np.clip(noisy_img, a_min=0, a_max=255)
    return noisy_img.astype(np.uint8)


def salt_and_pepper_noise(img, prob):
    """
    给图像添加椒盐噪声
    :param img: 输入的图像
    :param prob: 噪声的概率
    :return: 添加噪声后的图像
    """
    output = np.copy(img)
    noise = np.random.rand(*img.shape[:2])
    output[noise < prob / 2] = 0
    output[noise > 1 - prob / 2] = 255
    return output
