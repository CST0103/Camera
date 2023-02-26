## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
# 首先導入庫
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
# 導入 Numpy 以方便數組操作
import numpy as np
# Import OpenCV for easy image rendering
# 導入 OpenCV 方便圖像渲染
import cv2

from realsense_depth import *

# Create a pipeline
# 創建管道
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
# 創建配置並將管道配置為流式傳輸
#  different resolutions of color and depth streams
#  顏色和深度流的不同分辨率
config = rs.config()

# Get device product line for setting a supporting resolution
# 獲取設備產品線以設置支持分辨率
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

# Start streaming
# 開始流式傳輸
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
# (獲取深度傳感器的深度刻度（參見 rs-align 示例進行解釋）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
#  移除超過1公尺的背景
clipping_distance_in_meters = 0.75 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
#創建對齊對象
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)


# Streaming loop
# 傳輸循環
try:
    while True:
        # Get frameset of color and depth
        # 獲取顏色和深度的框架集
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        # 將深度框與顏色框對齊
        aligned_frames = align.process(frames)

        # Get aligned frames
        #獲取對齊的幀數
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        #驗證兩個幀都有效
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Remove background - Set pixels further than clipping_distance to grey
        #移除背景 - 將比 clipping_distance 更遠的像素設置為灰色
        grey_color = 0
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #渲染圖像:
        #   depth align to color on left
        #   深度與左側顏色對齊
        #   depth on right
        #   右邊的深度
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

        gray = cv2.medianBlur(gray, 5)

        rows = gray.shape[0]
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                  param1=100, param2=30,
                                  minRadius=1, maxRadius=80)

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(depth_colormap, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(depth_colormap, center, radius, (255, 0, 255), 3)

        images = np.hstack((bg_removed, depth_colormap))
        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()

