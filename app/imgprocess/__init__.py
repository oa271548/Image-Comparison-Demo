import cv2
import numpy as np
import imutils
from skimage.metrics import structural_similarity

def get_min_dimension(image1, image2):
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    return min(h1, h2), min(w1, w2), min(c1, c2)

'''
Compare two images and return the differences
'''
def image_compare_ssim(left_filename, right_filename):
    left_img = cv2.imread(left_filename)
    right_img = cv2.imread(right_filename)
    
    # Convert images to grayscale
    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    (score, diff) = structural_similarity(left_img_gray, right_img_gray, full=True)

    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")

    # Threshold the difference image
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Find contours to obtain the regions of the two input images that differ
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    diff_areas = []
    # Loop over each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 40:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            diff_areas.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
    
    return diff_areas

def image_compare_absdiff(left_filename, right_filename):
    left_img = cv2.imread(left_filename)
    right_img = cv2.imread(right_filename)

    # Convert images to grayscale
    left_img_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_img_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Find the difference between the two images using absdiff
    diff = cv2.absdiff(left_img_gray, right_img_gray)

    # Apply threshold
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Erosion
    kernel = np.ones((3, 3), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow("Erosion", erode)

    # Dilation
    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    cv2.imshow("Dilation", dilate)

    # Find contours to obtain the regions of the two input images that differ
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    diff_areas = []
    # Loop over each contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 60:
            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            diff_areas.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })

    return diff_areas
