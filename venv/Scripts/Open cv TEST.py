import cv2
from realsense_depth import *

dc = DepthCamera()
ret, depth_frame, color_frame = dc.get_frame()

cv2.imshow("Color_frame", color_frame)
cv2.waitKey(0)