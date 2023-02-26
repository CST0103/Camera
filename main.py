import cv2
import pyrealsense2
from realsense_depth import *
import sys
import cv2 as cv
import numpy as np

point = (0, 0)
center = (0, 0)
def show_distance(event, x, y, args, params):
    global point
    point = ( x, y)

# Initialize Camera Intel Realsense
dc = DepthCamera()

#Create mouse event

cv2.namedWindow("color_frame")
cv2.setMouseCallback("color_frame", show_distance)



for j in range(1,10) :

    ret, depth_frame, color_frame = dc.get_frame()



    gray = cv.cvtColor(color_frame, cv.COLOR_BGR2GRAY)

    gray = cv.medianBlur(gray, 5)

    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=150, param2=30,
                                  minRadius=1, maxRadius=20)

    if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:

                center = (i[0], i[1])
                # circle center
                cv.circle(color_frame, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv.circle(color_frame, center, radius, (255, 0, 255), 3)
                # Show distance for a specific point
                cv2.circle(color_frame, center, 4, (0, 0, 255))
                distance = depth_frame[center[1], center[0]]
                if distance == 0:
                    continue

                cv2.putText(color_frame, "{}mm".format(distance), (center[0], center[1]), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 0), 2)

                cv2.imshow("color_frame", color_frame)

cv2.waitKey(0)



