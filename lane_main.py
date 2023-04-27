#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
from module import pipeline,perspective_warp,sliding_window,get_curve,draw_lanes,get_steering_angle
import rospy
from std_msgs.msg import Int8MultiArray


# 시리얼 통신 설정
# 시리얼 포트와 전송할 데이터를 설정합니다.

rad = 0.5
speed = 1


def rad2deg(rad):
    return rad * 180 / math.pi

# Open the default camera (index 0)
cap = cv2.VideoCapture(0)

while True:

    # ros init
    pub = rospy.Publisher('SpeedAngleGear', Int8MultiArray, queue_size=10)
    rospy.init_node('talker', anonymous=False)
    rate = rospy.Rate(10) # 10hz


    data = Int8MultiArray()

    # Read a frame from the camera
    ret, frame = cap.read()

    dst = pipeline(frame)
    dst = perspective_warp(dst, dst_size=(640,480))

    # out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    out_img, curves, lanes, ploty = sliding_window(dst)
    left_curverad, right_curverad, center = get_curve(frame, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    # 각도와,조향각
    steering_angle = get_steering_angle(center)
    deg = -1 * rad2deg(steering_angle) - 8
    if abs(deg) >= 25 and deg < 0:
        deg = -25
    elif abs(deg) >= 25 and deg > 0:
        deg = 25
    
    data.data=[int(speed),int(deg),0]
    pub.publish(data)

    # final line image
    cv2.imshow('final image',img_)

    # steering angle and degree
    print(f'steering angle {steering_angle} real degree {deg}')

    # lane값
    # print(f'lane value {lanes}')
    # out_img -> sliding window image
    # cv2.imshow('sliding window',out_img) 

    # Wait for a key press (this allows the window to stay open)
    key = cv2.waitKey(1)
    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

# Release the camera and destroy the window
cap.release()
cv2.destroyAllWindows()