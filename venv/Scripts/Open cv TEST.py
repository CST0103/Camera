import cv2
import numpy as np

img = cv2.imread('distance test.jpg', 0)
gray = img

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
ret,thresh = cv2.threshold(blur_gray,100,255,0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(thresh, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 3  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 50  # minimum number of pixels making up a line
max_line_gap = 20 # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
squares = [[lines(1)]]

for line1 in lines:
    for square in squares:
        for line2 in square:
            if line1 != line2:
                if intersects(line1, line2):
                    square.append(line1)
                else:
                    squares.append([line1])

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
        print("x1 {} y1 {} x2 {} y2 {}".format(x1,y1,x2,y2))



lines_edges = cv2.addWeighted(img, 0.5, line_image, 1, 0)

line_image_gray = cv2.cvtColor(line_image, cv2.COLOR_RGB2GRAY)

M = cv2.moments(line_image_gray)

cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv2.circle(lines_edges, (cx, cy), 5, (0, 0, 255), 1)

cv2.imshow("res", lines_edges)
cv2.imshow("line_image", line_image)
cv2.waitKey(0)
cv2.destroyAllWindows()