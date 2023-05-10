import cv2
import numpy as np
from matplotlib import pyplot as plt
import iou
from watermark import embed_watermark_color, extract_watermark_color

"""
测试脚本 2
滤波
"""
if __name__ == '__main__':
    alpha = 0.8
    wm = cv2.imread('img/watermark/jmu_text.png', 0)
    img = cv2.cvtColor(cv2.imread('./img/GT/im5.png'), cv2.COLOR_BGR2RGB)

    embedded = embed_watermark_color(img, wm, alpha)
    extract_wm, _ = extract_watermark_color(embedded, wm.shape)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gauss_blur = cv2.GaussianBlur(embedded, (3, 3), 0, 0)
    img_mean = cv2.blur(embedded, (2, 2))
    high_pass = cv2.filter2D(embedded, -1, kernel)

    imgs = [img, embedded, gauss_blur, img_mean, high_pass]
    wms = [extract_watermark_color(imgs[i], wm.shape)[0] for i in range(5)]
    wms[0] = wm
    titles = ['原始图像', '嵌入水印', '高斯滤波', '均值滤波', '高通滤波']
    wm_iou = [iou.iou(wm, wms[i]) for i in range(5)]

    for i in range(1, 6):
        plt.subplot(2, 5, i), plt.imshow(imgs[i - 1]), plt.xticks([]), plt.yticks([])
        plt.title(titles[i - 1])
        plt.subplot(2, 5, i + 5), plt.imshow(wms[i - 1], cmap='gray'), plt.xticks([]), plt.yticks([])
        if i > 1:
            plt.title('提取水印 IOU: {:.3f}'.format(wm_iou[i - 1]))
        else:
            plt.title('水印')
plt.show()
