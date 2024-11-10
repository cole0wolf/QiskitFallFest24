import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def highlightStopSign(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)  # Approximation tolerance
        approx = cv2.approxPolyDP(contour, epsilon, True)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)

        if cv2.contourArea(contour) > 50:  # Filter by area
            if len(approx) == 8:
                #it is likely an 8 sided polygon
                if 0.5 < aspect_ratio < 2:  # assuming the object should have a certain aspect ratio range
                    # Highlight the object
                    cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)

        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
    
    # Display the result
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return image


inputDirectoryName = "outputImages"
outputDirectoryName = "highlightedImages"

directory = os.fsencode(inputDirectoryName)    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
        image = cv2.imread(inputDirectoryName + "/" + filename)
        cv2.imwrite(outputDirectoryName + "/" + str(filename), highlightStopSign(image))