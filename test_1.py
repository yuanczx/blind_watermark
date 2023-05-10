import os
import cv2
from matplotlib import pyplot as plt
import attack
from iou import iou
from watermark import embed_watermark_color, extract_watermark_color

if __name__ == '__main__':
    alpha = 0.8
    wm = cv2.imread('img/watermark/jmu_text.png', 0)
    psnr_result = []
    img_dir = 'img/MIT-Adobe_FiveK/'

    channel0_img = []
    merged_img = []
    channel0_iou = []
    merged_iou = []

    for jpg_name in os.listdir(img_dir):
        img = cv2.imread(img_dir + jpg_name)
        png_name = jpg_name.replace('jpg', 'png')
        embedded = embed_watermark_color(img, wm, alpha=alpha)
        cv2.imwrite('./out/embedded/start-alpha-' + str(alpha) + '_' + jpg_name, embedded)
        embedded_noisy = attack.gauss_noisy(embedded)
        extra_wm, channels = extract_watermark_color(embedded, wm.shape)

        channel0_img.append(channels[0])  # 通道0 iou
        merged_img.append(extra_wm)
        channel0_iou.append(iou(channels[0], wm))
        merged_iou.append(iou(extra_wm, wm))

        cv2.imwrite('./out/extract/start-alpha-' + str(alpha) + '_' + jpg_name, extra_wm)
        cv2.imwrite('./out/extract/start-alpha-' + str(alpha) + '_channel-0_' + jpg_name, channels[0])
        psnr = cv2.PSNR(img, embedded)

        plt.figure(), plt.subplot(2, 2, 1), plt.title('原始图像')
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb), plt.axis('off')

        plt.subplot(2, 2, 2), plt.title('嵌入水印后的图像\nα=0.8, PSNR: {:.2f}'.format(psnr))
        rgb = cv2.cvtColor(embedded, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb), plt.axis('off')

        wm_iou = iou(extra_wm, wm)
        plt.subplot(2, 2, 3), plt.title('水印'), plt.imshow(wm, cmap='gray'), plt.xticks([]), plt.yticks([])
        plt.subplot(2, 2, 4), plt.title('提取水印 IOU: {:.3f}'.format(wm_iou))
        plt.imshow(extra_wm, cmap='gray'), plt.xticks([]), plt.yticks([])

        print(jpg_name + ' PSNR:', psnr)
        psnr_result.append(psnr)

    plt.figure()
    for i in range(1, 6):
        plt.subplot(2, 5, i)
        plt.title('单通道[{i}] IOU: {:.3f}'.format(channel0_iou[i], i=i))
        plt.imshow(channel0_img[i], cmap='gray')
        plt.subplot(2, 5, i + 5)
        plt.title('合成[{i}] IOU: {:.3f}'.format(merged_iou[i], i=i))
        plt.imshow(merged_img[i], cmap='gray')
    plt.show()
