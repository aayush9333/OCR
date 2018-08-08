import cv2
import numpy as np
import pytesseract
from PIL import Image

#path folder
src_path = "C:/Python34/ocr/"

def get_string(img_path):

    img = cv2.imread(img_path)    #reading image

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #converting to grayscale

    #applying dialation
    kernel = np.ones((1,1),np.uint8)
    img = cv2.dilate(img,kernel,iterations=1)
    img = cv2.erode(img,kernel,iterations=1)

    #writing new image
    cv2.imwrite(src_path + "altered.png", img)

    #Apply threshold to get image with only black and white
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)                                            
    cv2.imwrite(src_path + "thres.png",img)

    result = pytesseract.image_to_string(Image.open(src_path + "thres.png"))

    return result

print ("---------------------------Start recognize text--------------------------------")

print (get_string(src_path + "ocr.png"))

print ("-----------------------------------Done----------------------------------------")
    
