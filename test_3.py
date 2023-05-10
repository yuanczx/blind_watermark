import cv2
from matplotlib import pyplot as plt
import attack
import iou
from watermark import embed_watermark_color, extract_watermark_color

"""
测试脚本3 
对图像加入不同噪声比对结果
"""


def add_noisy(img, wm, alpha):
    embedded = embed_watermark_color(img, wm, alpha)
    gauss_noisy = attack.gauss_noisy(embedded)
    poisson_noisy = attack.poisson_noisy(embedded)
    pepper_noisy = attack.salt_and_pepper_noise(embedded, 0.01)

    imgs = [img, embedded, gauss_noisy, poisson_noisy, pepper_noisy]
    wms = [extract_watermark_color(imgs[i], wm.shape)[0] for i in range(5)]
    wms[0] = wm
    wm_iou = [iou.iou(wms[i], wm) for i in range(5)]
    im_psnr = [cv2.PSNR(img, imgs[i]) for i in range(5)]
    titles = ['原始图像', '含水印图像\nPSNR:{:.2f}', '加入高斯噪声\n' + r'$\mu=0, \sigma=0.01$,PSNR:{:.2f}',
              '加入泊松噪声\nPSNR:{:.2f}', '加入椒盐噪声\nP=0.01, PSNR: {:.2f}']
    plt.figure()
    for i in range(5):
        plt.subplot(2, 5, i + 1), plt.imshow(imgs[i]), plt.yticks([]), plt.xticks([])
        plt.title(titles[i]) if i == 0 else plt.title(titles[i].format(im_psnr[i]))
        plt.subplot(2, 5, i + 6), plt.imshow(wms[i], cmap='gray'), plt.yticks([]), plt.xticks([])
        plt.title('水印') if i == 0 else plt.title('提取水印\nIOU: {:.3f}'.format(wm_iou[i]))


if __name__ == '__main__':
    alpha = 0.8
    wm = cv2.imread('img/watermark/jmu_text.png', 0)
    img = cv2.cvtColor(cv2.imread('./img/GT/im9.png'), cv2.COLOR_BGR2RGB)
    add_noisy(img, wm, alpha)
    img = cv2.cvtColor(cv2.imread('img/MIT-Adobe_FiveK/80.jpg'), cv2.COLOR_BGR2RGB)
    add_noisy(img, wm, alpha)
    plt.show()
