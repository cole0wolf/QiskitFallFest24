import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

def convertToEdgeMap(inputImg):
    assert img is not None, "file could not be read, check with os.path.exists()"
    edges = cv.Canny(img,100,200)

    #plt.subplot(121),plt.imshow(img,cmap = 'gray')
    #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

    return edges


directoryName = "RoadSigns/train/35"
directory = os.fsencode(directoryName)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    #if filename.endswith(".jpg"):
    img = cv.imread(directoryName + "/" + filename, cv.IMREAD_GRAYSCALE)
    cv.imwrite(directoryName + "/" + str(filename), convertToEdgeMap(img))
