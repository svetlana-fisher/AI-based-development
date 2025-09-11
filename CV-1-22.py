import cv2
import matplotlib.pyplot as plt
from skimage import data


def gaussian_blur():
    img = data.astronaut()
    roi = img[100:200, 150:250].copy()
    blurred_roi = cv2.GaussianBlur(roi, (15, 15), 0)
    blurred_img = img.copy()
    blurred_img[100:200, 150:250] = blurred_roi

    show_img(img=img, blurred_img=blurred_img, roi=roi, blurred_roi=blurred_roi)

def show_img(img, roi, blurred_img, blurred_roi):
    plt.figure(figsize=(15, 10))

    # original img
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')

    # ROI
    plt.subplot(2, 2, 2)
    plt.imshow(roi)
    plt.title('ROI')
    plt.axis('off')

    # blur ROI
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_roi)
    plt.title('ROI GaussianBlur')
    plt.axis('off')

    # blur img
    plt.subplot(2, 2, 4)
    plt.imshow(blurred_img)
    plt.title('GaussianBlur')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

gaussian_blur()