import cv2
import numpy as np
import pyrealsense2 as rs

# 定義攝像頭參數
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 15)

# 開始攝像頭串流
pipeline.start(config)

# 定義檢測函數
def detect_blocks(frame):
    # 轉換為灰度圖像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 應用閾值處理以分離積木和背景
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]

    # 應用輪廓檢測來檢測積木輪廓
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 迭代所有輪廓，找到積木輪廓並標記形狀
    for cnt in contours:
        # 過濾太小或太大的輪廓
        if cv2.contourArea(cnt) < 100 or cv2.contourArea(cnt) > 10000:
            continue

        # 適用輪廓逼近來獲取多邊形頂點
        approx = cv2.approxPolyDP(cnt, 0.05 * cv2.arcLength(cnt, True), True)

        # 根據多邊形頂點數標記形狀
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Rectangle"
        else:
            shape = "Circle"

        # 在圖像上標記形狀和位置
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame
while True:
    # 從攝像頭獲取影像
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()

    # 將影像轉換為OpenCV格式
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 檢測積木
    processed_image = detect_blocks(color_image)

    # 顯示影像
    cv2.imshow("Processed Image", processed_image)
    cv2.imshow("Color Image", color_image)

    # 按下“q”鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉所有視窗
cv2.destroyAllWindows()

# 停止攝像頭串流
pipeline.stop()
