#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import cv2 
import numpy as np
import pytesseract
from tensorflow import keras

#img = cv2.imread("D:\Reham\DATA\Graduation_project\OCR\data\wrap.jpg") # need to be replaced get_img function

#TODO: get_img function */
#TODO : Error Handleing -> opencv  */
#TODO: if not 14 num -> raise Exceptions (take another img) */ 
#TODO: imporve docstring  */
#TODO: divide functions */
#TODO: deepface functions 

def get_img_from_path(image_path):

    # ask how to take the image
    img = cv2.imread(image_path)
    if img is None:
        raise Exception("Image couldn't be read,try again")
    return img

    


def biggest_contour(contours):
    """get the rectangular contour of the ID picture

    Args:
        contours (list): a list of 10 biggest contours in ID pictures

    Returns:
        numpy.ndarray: a array of 4 points [[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]] 
        that make the rectangular outer shape of the ID
    """
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 25:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    return biggest
#-----------------------------------------------------------

def get_contours(img):
    """change img to an ID image 600*400

    Args:
        img (MatLike): the image
    Raises:
        ValueError: when contours list are empty
        ValueError: when biggest rectangular contours are empty
        Exception:  if Wrapping failed

    Returns:
        MatLike: the ID image after operations(contours,warp perspective,resizing to 600*400)
    """
    img = rotate_img(img,90)
    img_original = img.copy()

    # Image modification
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 20, 30, 30)
    edged = cv2.Canny(gray, 10, 20)

    # Contour detection
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    if contours is None:
        raise ValueError("Contours are empty")

    #get the rectangular contour
    biggest = biggest_contour(contours)
    if biggest.size == 0:
        raise ValueError("Empty rectangular contours")
     # cv2.drawContours(img, [biggest], -1, (0, 255, 0), 3)

    # Pixel values in the original image
    points = biggest.reshape(4, 2)
    input_points = np.zeros((4, 2), dtype="float32")

    points_sum = points.sum(axis=1)
    input_points[0] = points[np.argmin(points_sum)]
    input_points[3] = points[np.argmax(points_sum)]

    points_diff = np.diff(points, axis=1)
    input_points[1] = points[np.argmin(points_diff)]
    input_points[2] = points[np.argmax(points_diff)]

    (top_left, top_right, bottom_right, bottom_left) = input_points
    bottom_width = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + ((bottom_right[1] - bottom_left[1]) ** 2))
    top_width = np.sqrt(((top_right[0] - top_left[0]) ** 2) + ((top_right[1] - top_left[1]) ** 2))
    right_height = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + ((top_right[1] - bottom_right[1]) ** 2))
    left_height = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + ((top_left[1] - bottom_left[1]) ** 2))

    # Output image size
    max_width = max(int(bottom_width), int(top_width))
    max_height = max(int(right_height), int(left_height))
    #max_height = int(max_width * 1.414)  # for A4

    # Desired points values in the output image
    converted_points = np.float32([[0, 0], [max_width, 0], [0, max_height], [max_width, max_height]])

    # Perspective transformation
    matrix = cv2.getPerspectiveTransform(input_points, converted_points)
    img_output = cv2.warpPerspective(img_original, matrix, (max_width, max_height))
    if img_output is None:
        raise Exception("Wrapping failed")
    

    resized_image = cv2.resize(img_output, (600,400)) ## this is the one to be saved

    
    return resized_image

    #cv2.imwrite('data/wrap.output_resized.jpg', resized_image)
#------------------------------------------------------------------------


def resize_save(img,user_id):
    resized_image = cv2.resize(img, (600,400)) ## this is the one to be saved
    cv2.imwrite(f'data/output_resized{str(user_id)}.jpg', resized_image)
#---------------------------------------------------------------------
    
def crop_and_threshold(img):
    """crop the image to get the ID number then apply otsu's method to threshold

    Args:
        img (MatLike): ID image after contours,rotation ..etc

    Returns:
        MatLike: ID number image ready for OCR
    """
    
    # crop before threshold
    img_crop = img[315:359,265:599]

    gray_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    
    # Sharpen the image 
    kernel = np.array([[-1,-1,-1], [-1, 9,-1],[-1,-1,-1]]) 
    sharpened_image = cv2.filter2D(gray_img, -1, kernel)


    # threshold
    img_adaptive_threshold = cv2.adaptiveThreshold(sharpened_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,25) # 21,25
    ret , Otsu_thresh = cv2.threshold(gray_img,100,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    return img_adaptive_threshold
#------------------------------------------------------------



def OCR_pytesseract (img):
    """apply OCR on an image using pytesseract 

    Args:
        img (MatLike): ID number image

    Returns:
        str: ID number in English
    """

    custom_config = r'--oem 3 --psm 6'

    str_ID=pytesseract.image_to_string(img
    ,lang='ara_number',config=custom_config)
    #print(str_ID) #30110011255366
    return str_ID
#---------------------------------------------------------
def rotate_img(img_source,angle=270):
    """rotate image (if necessary)

    Args:
        img_source (MatLike): image to be rotated
        angle (int, optional): the angle to rotate by. Defaults to 270.
        
    Raises:
        ValueError: when rotation couldn't be preformed

    Returns:
        MatLike: Image rotated
    """
    height, width = img_source.shape[:2] 

    if (width<=height): 
        img_destination = img_source.copy() # don't rotate
        return img_destination
        
    angle = angle  # when the img is taken in a "portrait mode", it's rotated by 270
    center = (width/2, height/2)      # centre to be rotated about
    scale = 0.90
    rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)
    if rotation_matrix is None:
        raise ValueError("rotation couldn't be preformed")
    img_destination = cv2.warpAffine(src=img_source, M=rotation_matrix, dsize=(width, height)) 
    return img_destination
#------------------------------------------------
#------------------------------------------------
def OCR_pipline(img):
    """from ID img to ID number through applying contours and wrap prespective , cropping the image , thresholding and finally applying OCR

    Args:
        img (MatLike): ID image
        img_path(string): contains image path , defaults to ''
        
    Raises:
        Exception: if ID_number are not  14 digit

    Returns:
        int: ID number
    """

    contoured_pic = get_contours(img)  
    ID_pic = crop_and_threshold(contoured_pic)
    string_OCR = OCR_pytesseract(ID_pic).replace(" ","").replace("\n","").rstrip()
    National_ID = int(string_OCR)

    if len(string_OCR) != 14:
        raise Exception("Photo is unclear,upload a new photo")
    
    return National_ID 
#-------------------------------------------------
#Second part: deepface library




# In[ ]:

