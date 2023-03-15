import cv2
import pyrealsense2
from realsense_depth import *
import sys
import cv2 as cv
import numpy as np

def draw_min_rect_circle(img, cnts):
    img = np.copy(img)
    for cnt in cnts:
        min_rect = cv2.minAreaRect(cnt)
        min_rect = cv2.boxPoints(min_rect)
        min_rect = min_rect.astype(np.int32)
        cv2.drawContours(img, [min_rect], -1, (255, 0, 0), 2)
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        img = cv2.circle(img, center, radius, (0, 0, 255), 2)
    return img


def draw_approx_hull_polygon(img, cnts):
    img = np.copy(img)
    for cnt in cnts:
        hull = cv2.convexHull(cnt)
        epsilon = 0.01 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        cv2.drawContours(img, [approx], -1, (0, 0, 255), 2)
    return img


def draw_contours(img, cnts):  # conts = contours
    img = np.copy(img)
    img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
    return img


def run():
    capture = cv2.VideoCapture(0)

    while True:
        ret, image = capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        canny = cv2.Canny(thresh, 128, 256)

        contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        imgs = [
            image, thresh,
            draw_min_rect_circle(image, contours),
            draw_approx_hull_polygon(image, contours),
        ]

        for img in imgs:
            cv2.imshow("contours", img)

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
