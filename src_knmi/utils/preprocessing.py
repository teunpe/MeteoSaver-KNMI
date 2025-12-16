import cv2
import matplotlib.pyplot as plt
# from src.table_and_cell_detection_model import deskew
import numpy as np
from cv2.typing import MatLike

def deskew(image: MatLike):
    """Rotate the image to correct skew. Using OpenCV's findContours to find the largest 
    contour, take the minimum area rectangle, and then rotate the image to correct the skew.

    Parameters
    ----------
    image
        _description_

    Returns
    -------
        _description_
    """    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load the image
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours:
    # # Assume the largest contour is the table
    # table_contour = contours[1]

        # Find the minimum area rectangle
        rect = cv2.minAreaRect(contour)
        angle = rect[-1]
        # print(angle)
        if angle > 10 or angle < -10:
            # Rotate the image  
            if angle < -87:
                angle += 90
                break
            elif angle > 87:
                angle -= 90
                break
            print(f"Angle {angle:.2f} too large, trying next contour")
            continue
        break

    # Rotate the image  
    # if angle < -45:
    #     angle += 90
    # elif angle > 45:
    #     angle -= 90
    print(f"Rotating image by {angle:.2f} degrees")

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
    return rotated, M

def read_img(image_path: str) -> MatLike:
    # Read the image
    image = cv2.imread(image_path)
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_preprocessing(image: MatLike, cutoff_hsv=100, skew=True, blur=False, blocksize=51, C=6):
    """Get grayscale and binarized image, optionally rotated to correct skew.

    Parameters
    ----------
    image
        binarized or grayscale image
    cutoff_hsv, optional
        by default 100
    skew, optional
        by default True

    Returns
    -------
        grayscale image, binarized image, original image
    """    
    ## Read Image from the given image path
    original_image  = image
    if skew:
        original_image, rotation_matrix = deskew(original_image)
    # Convert image to grayscale
    if blur:
        original_image = cv2.GaussianBlur(original_image, (3, 3), 0)
    image_in_grayscale = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_RGB2HSV)

    # Define a range of HSV values to mask the paper
    # These values need to be adjusted based on your specific image
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([cutoff_hsv, cutoff_hsv, cutoff_hsv])

    # Create a mask using the HSV range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    binarized_image = cv2.bitwise_not(mask)

    image_in_grayscale = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    # binarized_image = cv2.adaptiveThreshold(image_in_grayscale,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #         cv2.THRESH_BINARY,201,2)
    # _, binarized_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(image_in_grayscale,(5,5),0)
    # _, binarized_image = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    binarized_image = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,blockSize=blocksize,C=C)

    # C: subtraction that gets rid of some noise
    # blockSize: size of the neighborhood area used to calculate the threshold value
    return image_in_grayscale, binarized_image, original_image
