import cv2
import matplotlib.pyplot as plt
from skimage import data
from jsonargparse import CLI
import numpy as np


def gaussian_blur(
    h_start: int=100, 
    h_end: int=200,
    w_start: int=150, 
    w_end: int=250,
    kernel_size: int=15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    h_start : int, optional
        Starting height coordinate for ROI (y-start), by default 100
    h_end : int, optional
        Ending height coordinate for ROI (y-end), by default 200
    w_start : int, optional
        Starting width coordinate for ROI (x-start), by default 150
    w_end : int, optional
        Ending width coordinate for ROI (x-end), by default 250
    kernel_size : int, optional
        Size of Gaussian kernel (must be odd), by default 15
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns tuple containing:
        - img: original image
        - blurred_img: image with blurred ROI
        - roi: original ROI region
        - blurred_roi: blurred ROI region

    """

    img = data.astronaut()
    if h_start >= h_end or w_start >= w_end:
        raise ValueError("Invalid ROI coordinates: start must be less than end")
    if h_end > img.shape[0] or w_end > img.shape[1]:
        raise ValueError("ROI coordinates exceed image dimensions")
    
    try:
        roi = img[h_start:h_end, w_start:w_end].copy()
    except ValueError as e:
        raise ValueError(f"Invalid value. Error: {e}")
    
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    blurred_img = img.copy()
    blurred_img[h_start:h_end, w_start:w_end] = blurred_roi
    return img, blurred_img, roi, blurred_roi


def show_img(
    img: np.ndarray, 
    roi: np.ndarray, 
    blurred_img: np.ndarray, 
    blurred_roi: np.ndarray, 
    figsize: tuple[int, int] = (15, 10)
) -> None:
    """
    Parameters
    ----------
    img : np.ndarray
        Original input image
    roi : np.ndarray
        Region of interest extracted from the image
    blurred_img : np.ndarray
        Image with blurred ROI applied
    blurred_roi : np.ndarray
        Blurred region of interest
    figsize : Tuple[int, int], optional
        Figure size for matplotlib plot, by default (15, 10)

    """

    plt.figure(figsize=figsize)

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

def main(
    h_start: int=100, 
    h_end: int=200,
    w_start: int=150, 
    w_end: int=250,
    kernel_size: int=15,
):
    """
    Parameters
    ----------
    h_start : int, optional
        Starting height coordinate for ROI, by default 100
    h_end : int, optional
        Ending height coordinate for ROI, by default 200
    w_start : int, optional
        Starting width coordinate for ROI, by default 150
    w_end : int, optional
        Ending width coordinate for ROI, by default 250
    kernel_size : int, optional
        Size of Gaussian kernel, by default 15

    """

    img, blurred_img, roi, blurred_roi = gaussian_blur(h_start=h_start, h_end=h_end, w_start=w_start, w_end=w_end, kernel_size=kernel_size
    )
    show_img(img=img, roi=roi, blurred_img=blurred_img, blurred_roi=blurred_roi)


CLI(main)
