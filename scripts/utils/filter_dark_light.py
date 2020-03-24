import cv2
import numpy as np


def filter_dark_light_image(img, threshold=(30, 220), pixel_thresh=(0.88, 0.75), show_hist=False):
    pixels_num = img.shape[0] * img.shape[1] * img.shape[2]
    bgr_planes = cv2.split(img)
    hist_size = 256
    hist_range = (0, 256)
    accumulate = False

    b = np.squeeze(cv2.calcHist(bgr_planes, [0], None, [hist_size], hist_range, accumulate=accumulate))
    g = np.squeeze(cv2.calcHist(bgr_planes, [1], None, [hist_size], hist_range, accumulate=accumulate))
    r = np.squeeze(cv2.calcHist(bgr_planes, [2], None, [hist_size], hist_range, accumulate=accumulate))
    bgr = np.concatenate([np.expand_dims(b, axis=0), np.expand_dims(g, axis=0), np.expand_dims(r, axis=0)])

    dark = np.sum(bgr[:, :threshold[0]])
    light = np.sum(bgr[:, threshold[1]:])
    dark_pixel_threshold = pixels_num * pixel_thresh[0]
    light_pixel_threshold = pixels_num * pixel_thresh[1]
    if show_hist:
        draw_histogram(img, hist_size, b, g, r)
    if dark > dark_pixel_threshold:
        return 1
    elif light > light_pixel_threshold:
        return 2
    else:
        return 0


def draw_histogram(img, hist_size, b, g, r):
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / hist_size))
    hist_image = np.zeros((hist_h, hist_w, 3), dtype=np.uint8)

    cv2.normalize(b, b, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(g, g, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(r, r, alpha=0, beta=hist_h, norm_type=cv2.NORM_MINMAX)

    for i in range(1, hist_size):
        cv2.line(hist_image, (bin_w * (i - 1), hist_h - int(round(b[i - 1]))),
                (bin_w * i, hist_h - int(round(b[i]))), (255, 0, 0), thickness=2)
        cv2.line(hist_image, (bin_w * (i - 1), hist_h - int(round(g[i - 1]))),
                (bin_w * i, hist_h - int(round(g[i]))), (0, 255, 0), thickness=2)
        cv2.line(hist_image, (bin_w * (i - 1), hist_h - int(round(r[i - 1]))),
                (bin_w * i, hist_h - int(round(r[i]))), (0, 0, 255), thickness=2)

    cv2.imshow('Source image', img)
    cv2.imshow('calcHist', hist_image)
    cv2.waitKey()


def main():
    image = '/home/dmitry/virtualenv/mcmt_extra/Camera_Tampering_patched/Camera_Tampering/0012631/0012631_c0s0_000008400_01.jpg'
    img = cv2.imread(image)
    dl = filter_dark_light_image(img, show_hist=True)


if __name__ == '__main__':
    main()