#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Int8MultiArray
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import rospy
from std_msgs.msg import Int8MultiArray


def img_white(image):
    # BGR 제한 값 설정
    bgr_lower = np.array([240, 240, 240])
    bgr_upper = np.array([255, 255, 255])

    # HSV 제한 값 설정
    hsv_lower = np.array([70, 0, 200])
    hsv_upper = np.array([180, 50, 255])

    # BGR 공간에서 흰색 영역 찾기
    mask_bgr = cv2.inRange(image, bgr_lower, bgr_upper)

    # HSV 공간에서 흰색 영역 찾기
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_hsv = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # BGR, HSV 마스크를 합침
    mask = cv2.bitwise_or(mask_bgr, mask_hsv)

    # 마스크를 이용하여 원본 이미지에서 흰색 영역 추출
    white = cv2.bitwise_and(image, image, mask=mask)

    return white

def region_of_interest(img, color3=(255,255,255), color1=255): # ROI 셋팅
    # 480,640
    height, width = img.shape[0],img.shape[1]
    # 좌아래 좌상단 우상단 우아래
    vertices = np.array([[(0,height),(0, height//2+40), (width, height//2+40), (width,height)]], dtype=np.int32)
    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지
    
    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1
        
    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움 
    cv2.fillPoly(mask, vertices, color)
    
    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image

def pipline(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gaussian
    blur_img = cv2.GaussianBlur(gray_img,(3,3),0)
    # 이미지의 hough space의 교차점이 증가할때마다 +1을 transform에서 해줌
    # 근데 이를 교차점을 누적하고 threshold값을 넘기면 직선이라고 판단하기 위해 사용
    # 이 값이 높으면 직선이라 판단
    low_threshold, high_threshold = 70,210
    canny = cv2.Canny(blur_img,low_threshold,high_threshold)

    return canny

def draw_lines(img, lines, color=[0, 0, 255], thickness=10): # 선 그리기
    lines_image = np.zeros_like(img)
    cross_list = []
    if lines is not None:
        for line in lines:
            for idx,val in enumerate(line):
                x1,y1,x2,y2 = val
                if np.degrees(np.arctan(abs(y1-y2)/abs(x1-x2))) < 10:
                    # 왼쪽 라인
                    if idx == 0:
                        x1,x2 = 0,10
                    # 오른쪽 라인
                    else:
                        x1,x2 = img.shape[1],img.shape[1]-10
                cross_list.append(x2)
                cv2.line(lines_image, (x1, y1), (x2, y2), color, thickness)
    x = (cross_list[0]+cross_list[1])//2
    origin_x,origin_y = img.shape[1]//2,int(img.shape[0]*(2/5))
    center_sub = x - origin_x
    if center_sub == 0:
        center_sub = 1
    rad = np.arctan(origin_y/center_sub)
    angle_degrees = np.degrees(rad)

    if center_sub < 0:
        print('왼쪽으로가시오',angle_degrees)
    else:
        print('오른쪽',angle_degrees)

    cv2.circle(img,(x,int(img.shape[0]*(3/5))),2,(0,23,80),4)

    return lines_image,angle_degrees

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap): # 허프 변환
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    return lines

def weighted_img(img, initial_img, a=0.8, b=1., c=1.): # 두 이미지 operlap 하기
    return cv2.addWeighted(initial_img, a, img, b, c)
    

def make_coordinates(image, line_parameters,dir):
    slope, intercept = line_parameters
    if slope == 0:
        slope = 1
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1- intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    if x1 > 10000 or x2 > 10000 or x1 < -10000 or x2 < -10000:
        if dir == 'right':
            x1 = image.shape[1]
            x2 = image.shape[1]-10
        elif dir == 'left':
            x1 = 0
            x2 = 10

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # 다항식 만드는코드로서 직선에 대한 기울기의 파라미터를 구함
        parameter = np.polyfit((x1, x2), (y1, y2), 1)
        # 기울기와 y = slope * x + intercept
        slope = parameter[0]
        intercept = parameter[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    # (slop_avg,inter_avg) slope 단위로 평균낸것과, intercept평균
    if right_fit:
        right_fit_average = np.average(right_fit, axis =0)
        right_line = make_coordinates(image, right_fit_average,'right')
    else:
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = image.shape[1]
        x2 = x1 - 10
        right_line = np.array([x1,y1,x2,y2])
    if left_fit:
        left_fit_average =np.average(left_fit, axis=0)
        left_line =make_coordinates(image, left_fit_average,'left')
    else:
        y1 = image.shape[0]
        y2 = int(y1*(3/5))
        x1 = 0
        x2 = x1 + 10
        right_line = np.array([x1,y1,x2,y2])

    return np.array([[left_line, right_line]])

speed = 1

# ROS callback 함수
def image_callback(ros_image):
    global bridge

    # ROS 이미지 메시지를 OpenCV 이미지로 변환
    # 360 640 3
    frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
    lanelines_image = frame.copy()
    white_img = img_white(frame)
    dst = pipline(white_img)
    roi_img = region_of_interest(dst)
    lines = hough_lines(roi_img, 1, 1 * np.pi/180, 30, 10, 20)
    averaged_lines = average_slope_intercept(lanelines_image, lines) 
    lines_image,angle_degree = draw_lines(lanelines_image, averaged_lines)
    # 차선 검출한 부분을 원본 image에 overlap 하기
    result_img = weighted_img(lanelines_image,lines_image)

    cv2.imshow('original', result_img)
 
    
    data.data=[int(speed),int(angle_degree),0]
    pub.publish(data)

    # final line image
    cv2.imshow('final image',result_img)

    # steering angle and degree
    print(f'steering angle {angle_degree}')

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
    # rospy.Subscriber("/zed2i/zed_node/right_raw/image_raw_color", Image, image_callback)

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
