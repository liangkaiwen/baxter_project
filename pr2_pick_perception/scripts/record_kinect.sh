#!/bin/bash
rosbag record /head_mount_kinect/depth_registered/points /tf /head_mount_kinect/depth_registered/camera_info /head_mount_kinect/rgb/camera_info /head_mount_kinect/rgb/image_rect_color /head_mount_kinect/depth_registered/image_raw /pr2_pick_visualization
