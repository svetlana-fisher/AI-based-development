import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from jsonargparse import CLI
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_roi_coordinates(
    img_shape: tuple[int, int, int], 
    h_start: int, 
    h_end: int, 
    w_start: int, 
    w_end: int
) -> None:
    """
    Parameters
    ----------
    img_shape : tuple[int, int, int]
        Shape of the image (height, width, channels)
    h_start : int
        Starting height coordinate for ROI
    h_end : int
        Ending height coordinate for ROI
    w_start : int
        Starting width coordinate for ROI
    w_end : int
        Ending width coordinate for ROI
    
    """
    height, width = img_shape[:2]
    
    if h_start >= h_end or w_start >= w_end:
        raise ValueError("Invalid ROI coordinates: start must be less than end")
    
    if h_start < 0 or w_start < 0:
        raise ValueError("ROI coordinates cannot be negative")
    
    if h_end > height or w_end > width:
        raise ValueError(f"ROI coordinates exceed image dimensions. "
                        f"Image size: {width}x{height}, "
                        f"ROI: {w_start}:{w_end}, {h_start}:{h_end}")


def validate_kernel_size(kernel_size: int) -> None:
    """
    Parameters
    ----------
    kernel_size : int
        Size of Gaussian kernel

    """
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    if kernel_size > 31:  # Reasonable limit for Gaussian blur
        logger.warning("Large kernel size (%d) may cause performance issues", kernel_size)


def load_image(img_path: Path | None = None) -> np.ndarray:
    """
    Parameters
    ----------
    img_path : Path | None
        Path to image file, by default None (uses sample image)
    
    Returns
    -------
    np.ndarray
        Loaded image in RGB format

    """
    if img_path is None:
        img = data.astronaut()
        return img
    
    if not img_path.exists():
        raise FileNotFoundError(f"File '{img_path}' not found")
    
    if not img_path.is_file():
        raise ValueError(f"'{img_path}' is not a file")
    
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image '{img_path}'. "
                         "Check file format and integrity.")
    
    return img


def gaussian_blur(
    img_path: Path | None = None,
    h_start: int = 100, 
    h_end: int = 200,
    w_start: int = 150, 
    w_end: int = 250,
    kernel_size: int = 15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    img_path : Path | None
        Path to image file, by default None (uses sample image)
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
    validate_kernel_size(kernel_size)
    img = load_image(img_path)
    validate_roi_coordinates(img.shape, h_start, h_end, w_start, w_end)
    
    try:
        roi = img[h_start:h_end, w_start:w_end].copy()
        
        # Apply Gaussian blur to ROI
        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
        
        # Create result image with blurred ROI
        blurred_img = img.copy()
        blurred_img[h_start:h_end, w_start:w_end] = blurred_roi
        
        return img, blurred_img, roi, blurred_roi
        
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")


def show_img(
    img: np.ndarray, 
    roi: np.ndarray, 
    blurred_img: np.ndarray, 
    blurred_roi: np.ndarray, 
    figsize: tuple[int, int] = (12, 8)
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
    figsize : tuple[int, int], optional
        Figure size for matplotlib plot, by default (12, 8)
    """
    # Prepare data for plotting
    images_data = [
        (img, 'Original Image'),
        (roi, 'Region of Interest (ROI)'),
        (blurred_roi, 'Blurred ROI'),
        (blurred_img, 'Image with Blurred ROI')
    ]
    
    plt.figure(figsize=figsize, facecolor='white')
    
    for i, (image, title) in enumerate(images_data, 1):
        plt.subplot(2, 2, i)
        plt.imshow(image)
        plt.title(title, fontsize=12, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main(
    img_path: Path | None = None,  
    h_start: int = 100, 
    h_end: int = 200,
    w_start: int = 150, 
    w_end: int = 250,
    kernel_size: int = 15,
    show_result: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Parameters
    ----------
    img_path : Path, optional
        Path to image file, by default None (uses sample image)
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
    show_result : bool, optional
        Whether to display results, by default True
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
        Processing results if show_result is False, None otherwise
    """
    try:
        # Process image
        results = gaussian_blur(
            img_path=img_path, 
            h_start=h_start, 
            h_end=h_end, 
            w_start=w_start, 
            w_end=w_end, 
            kernel_size=kernel_size
        )
        
        img, blurred_img, roi, blurred_roi = results
        
        # Display results if requested
        if show_result:
            show_img(img=img, roi=roi, blurred_img=blurred_img, blurred_roi=blurred_roi)
            return None
        else:
            return results
            
    except FileNotFoundError as e:
        logger.error("File error: %s", e)
        raise
    except ValueError as e:
        logger.error("Validation error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


if __name__ == "__main__":
    CLI(main, description="Apply Gaussian blur to a region of interest in an image")