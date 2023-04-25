"""_summary_  
This code to capture the dominant colour and shape in the image.
It is more suitable for one dominant colour.
How it's works
#install library call colorthief
#install matplotlib
#give the Image location
"""


import cv2
from colorthief import ColorThief
import matplotlib.pyplot as plt


def detect_shapes_and_color(image):
    image_path = image
    image = cv2.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image
    _, thresh_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for i, contour in enumerate(contours):
        if i == 0:
            continue
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(image, contour, 0, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(approx)
        x_mid = int(x + (w / 3))
        y_mid = int(y + (h / 1.5))
        coords = (x_mid, y_mid)
        shape_label = "Triangle" if len(approx) == 3 else "Circle"
        cv2.putText(image, shape_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # dominant color
    ct = ColorThief(image_path)
    dominant_color = ct.get_color(quality=1)
    cv2.rectangle(image, (10, 50), (110, 80), dominant_color, -1)
    cv2.putText(image, "Dominant Color", (120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Display the image with detected shapes and dominant color
    cv2.imshow("Sunglasses Shape and Color Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_shapes_and_color(image="sun3.jpg")
