import cv2

# 選擇外接的webcam
cap = cv2.VideoCapture(0)

c = 1
timeF = 10  # frame time

while (1):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    if (c % timeF == 0 or c % timeF == 5):  # frame 限制

        dst = cv2.pyrMeanShiftFiltering(frame, 10, 50)  # 濾波
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)  # 灰度
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)  # 二值化

        cv2.imshow("ShiftFiltering", dst)
        cv2.imshow("threshold", thresh)

        image, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for i in cnts:

            # 輪廓近似
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.01 * peri, True)

            if len(approx) == 4:
                start_point = (approx[0][0][0], approx[0][0][1])
                end_point = (approx[2][0][0], approx[2][0][1])
                color = (255, 0, 0)
                image = cv2.rectangle(frame, start_point, end_point, color, 2)
                cv2.imshow("result", image)

    c = c + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        final = image
        break

# 釋放攝影機
cap.release()
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
cv2.imshow("final", final)
cv2.waitKey(0)