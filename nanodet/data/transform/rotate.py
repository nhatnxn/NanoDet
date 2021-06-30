import random
import numpy as np
import cv2
import math

def rotate(image, angle=180, scale=1.0):
    w = image.shape[1]
    h = image.shape[0]

    #rotate matrix
    ang = random.uniform(-angle,angle)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    
    #rotate
    image = cv2.warpAffine(image, M, (w,h))

    return image
