#!/usr/bin/env python3

import os
import rospy
import math
import random
import cv2
from typing import Union
import numpy as np
import time

from duckietown_msgs.msg import Twist2DStamped
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage
from duckietown.dtros import DTROS, NodeType, TopicType

class MySubscriberNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        # construct publisher
        self._speed_gain = 0.41
        self._steer_gain = 8.3
        self._bicycle_kinematics = 0.0
        self._simulated_vehicle_length = 0.18

        self.__bridge = CvBridge()
        self._last_detect = 0.0

        # Publications
        self.pub_car_cmd = rospy.Publisher(
            "/"+ os.environ['VEHICLE_NAME'] + "/joy_mapper_node/car_cmd", Twist2DStamped, queue_size=1, dt_topic_type=TopicType.CONTROL
        )

        self.__image_sub = rospy.Subscriber('/'+ os.environ['VEHICLE_NAME'] + "/camera_node/image/compressed", CompressedImage, self.callback)
        

    def callback(self, data):
        if time.time() - self._last_detect < 0.6:
            return
        self._last_detect = time.time()
        #self._last_detect = 10 * time.time()
        image = self.__bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        #rate = rospy.Rate(1) # 1Hz
        detected = self.detect_aruco_position(image)
        rotation = 0.0
        speed = 0.0
        
        if detected is None:
            rotation = 0.75
        elif detected['rotation_cmd'] == 0:
            speed = 1.0 if detected['distance_cmd'] else 0.0
        else:
            rotation = 0.75 * detected['rotation_cmd']
        self.pub_speed_rot(speed, rotation)

    def pub_speed_rot(self, speed, rotation, interval = 0.3):
        car_cmd_msg = Twist2DStamped()
        car_cmd_msg.header.stamp = rospy.get_rostime()
        # Left stick V-axis. Up is positive
        car_cmd_msg.v = speed * self._speed_gain
        if self._bicycle_kinematics:
            # Implements Bicycle Kinematics - Nonholonomic Kinematics
            # see https://inst.eecs.berkeley.edu/~ee192/sp13/pdf/steer-control.pdf
            steering_angle = rotation * self._steer_gain
            car_cmd_msg.omega = car_cmd_msg.v / self._simulated_vehicle_length * math.tan(steering_angle)
        else:
            # Holonomic Kinematics for Normal Driving
            car_cmd_msg.omega = rotation * self._steer_gain
        self.pub_car_cmd.publish(car_cmd_msg)
        time.sleep(interval)
        car_cmd_msg.v = 0.0
        car_cmd_msg.omega = 0.0
        self.pub_car_cmd.publish(car_cmd_msg)

    def detect_aruco_position(self, image: np.ndarray) -> Union[list, None]:
        (h, w) = image.shape[:2]
        SENSITIVITY_X = w // 5
        SENSITIVITY_Y = w // 7
        aruco_dict = cv2.aruco.getPredefinedDictionary( cv2.aruco.DICT_6X6_250 )
        aruco_params = cv2.aruco.DetectorParameters()
        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

        if len(corners) == 0:
            return None
        
        corner = corners[0][0]
        # print("Corners: ",corners)
        marker_x = np.sum(corner[:, 0]) / 4
        command = {'rotation': 0, 'distance': 0, 'distance_cmd': 0, 'rotation_cmd': 0}
        image_x_center = w / 2
        if abs(image_x_center - marker_x) < SENSITIVITY_X:
            command['rotation_cmd'] = 0
        elif image_x_center - marker_x < 0:
            command['rotation_cmd'] = -1
        elif image_x_center - marker_x > 0:
            command['rotation_cmd'] = 1
        command['rotation'] = image_x_center - marker_x
        
        # Use y values to detect the marker dimensions as they will be not distorted by any rotation
        y_coordinates = list(corner[:, 1])
        y_coordinates.sort()
        #print(y_coordinates)
        marker_real_height = abs(y_coordinates[0] - y_coordinates[3])
        #marker_real_height = 0
        command['distance_cmd'] = marker_real_height < SENSITIVITY_Y
        command['distance'] = marker_real_height
        return command



if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='aruco_marker_follower_node')
    rospy.spin()

