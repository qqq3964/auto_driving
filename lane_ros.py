#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
from module import pipeline,perspective_warp,sliding_window,get_curve,draw_lanes,get_steering_angle

rad = 0.5
speed = 1

def rad2deg(rad):
    return rad * 180 / math.pi

# ROS callback 함수
def image_callback(ros_image):
    global bridge

    # ROS 이미지 메시지를 OpenCV 이미지로 변환
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")

    frame = cv2.resize(frame,(1280,720))


    dst = pipeline(frame)
    dst = perspective_warp(dst, dst_size=(1280,720))

    # out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty
    out_img, curves, lanes, ploty = sliding_window(dst)

    left_curverad, right_curverad, center = get_curve(frame, curves[0],curves[1])
    img_ = draw_lanes(frame, curves[0], curves[1])
    
    # 각도와,조향각
    steering_angle = get_steering_angle(center)
    deg = -1 * rad2deg(steering_angle)
    
    if abs(deg) >= 25 and deg < 0:
        deg = -25
        
    elif abs(deg) >= 25 and deg > 0:
        deg = 25
    
    data.data=[int(speed),int(deg),0]
    pub.publish(data)

    # final line image
    cv2.imshow('final image',img_)
    cv2.imshow('sliding image',out_img)

    # steering angle and degree
    print(f'steering angle {steering_angle} real degree {deg}')

    # Wait for a key press (this allows the window to stay open)
    key = cv2.waitKey(1)
    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        rospy.signal_shutdown("KeyboardInterrupt")

if __name__ == '__main__':
    rospy.init_node('image_subscriber', anonymous=True)

    # 이미지를 ROS에서 OpenCV로 변환하기 위한 CvBridge 객체 생성
    bridge = CvBridge()

    # ROS 이미지 토픽을 구독하고 콜백 함수 등록
    rospy.Subscriber("/usb_cam/image_raw", Image, image_callback)

    # 시리얼 통신 설정
    # 시리얼 포트와 전송할 데이터를 설정합니다.
    pub = rospy.Publisher('SpeedAngleGear', Int8MultiArray, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    data = Int8MultiArray()

    # ROS 루프 실행
    while not rospy.is_shutdown():
        rospy.spin()

    # Release the camera and destroy the window
    cv2.destroyAllWindows()
