import os
import cv2
import requests
import numpy as np

for imagePath in os.listdir('DataSet'):
    
    if int(imagePath[0]) == 1:
        print(imagePath)
    
