#!/usr/bin/env python


import os
import sys

import os.path
import os
import math
import random
import numpy as np
import networkx as nx
import traceback
import roslib
import rospy
import std_msgs
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Quaternion
from sensor_msgs.msg import PointCloud2
import tf.msg
import geometry_msgs.msg
import tf
import threading
import subprocess
import baxter_external_devices
import connor_util.connor_util as cutil
import connor_util.static_publisher as static_pub
import pointclouds

SHARE_FOLDER = cutil.findFile("share")
SAVE_FILE = os.path.join(SHARE_FOLDER, "alignment_save.txt")
POINTS_FILE = os.path.join(SHARE_FOLDER, "alignment_points.txt")
LEFT_ANCHOR = dict(position=Point(x=0, y=0, z=0.01), frame_id="/left_hand", color=(1,1,0,1))
RIGHT_ANCHOR = dict(position=Point(x=0, y=0, z=0.01), frame_id="/right_hand", color=(1,1,0,1))

def load_points():
    if not os.path.exists(POINTS_FILE):
        return []
    ret = []
    with open(POINTS_FILE) as f:
        for line in f.read().split('\n'):
            if len(line) <= 0:
                continue
            vals = [float(x) for x in line.split(',')]
            ret.append((Point(x=vals[0], y=vals[1], z=vals[2]), Point(x=vals[3], y=vals[4], z=vals[5])))
    if len(ret) > 0:
        LEFT_ANCHOR['position'] = ret[-1][0]
        RIGHT_ANCHOR['position'] = ret[-1][1]
        del ret[-1]
    return ret
    
def save_points(alignment_points):
    with open(POINTS_FILE, "w") as f:
        #for p1, p2 in alignment_points + [(LEFT_ANCHOR['position'], RIGHT_ANCHOR['position'])]:
        for p1, p2 in [(LEFT_ANCHOR['position'], RIGHT_ANCHOR['position'])]:
            f.write(str(p1.x) + ",")
            f.write(str(p1.y) + ",")
            f.write(str(p1.z) + ",")
            f.write(str(p2.x) + ",")
            f.write(str(p2.y) + ",")
            f.write(str(p2.z) + "\n")
        f.flush()

def load_settings():
    if not os.path.exists(SAVE_FILE):
        return [0.22, 0.04, 0.78, 0.1, 0.87, 0.0, 1.0]
    with open(SAVE_FILE) as f:
        ret = map(float, f)
    return ret
    
def save_settings(a):
    with open(SAVE_FILE, "w") as f:
        f.write("\n".join(map(str, a)))

def load_settings():
    if not os.path.exists(SAVE_FILE):
        return [0.22, 0.04, 0.78, 0.1, 0.87, 0.0, 1.0]
    with open(SAVE_FILE) as f:
        ret = map(float, f)
    return ret
    
def save_settings(a):
    with open(SAVE_FILE, "w") as f:
        f.write("\n".join(map(str, a)))
        
def create_marker_array(*points):
    markerArray = MarkerArray()
    axis_length = 0.1
    for i in range(3):
        marker = Marker()
        marker.header.frame_id = "/camera_rgb_optical_frame"
        marker.ns = "alignment_markers"
        marker.id = len(markerArray.markers)
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.color.a = 1.0     
        marker.scale.x = 0.005
        marker.scale.y = 0.005
        marker.scale.z = axis_length   
     
        if i == 0: #red
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.position = Point(x=axis_length/2, y=0, z=0)
            marker.pose.orientation = cutil.eulerToQuaternion((math.pi/2,0,0))
        elif i == 1: #green
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.pose.position = Point(x=0, y=axis_length/2, z=0)
            marker.pose.orientation = cutil.eulerToQuaternion((0,0,math.pi/2))
        elif i == 2: #blue
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.pose.position = Point(x=0, y=0, z=axis_length/2)
            marker.pose.orientation = cutil.eulerToQuaternion((0,0,0))
        markerArray.markers.append(marker)
        
    for p in points:
        if type(p) is Point:
            p = dict(position=p)
        marker = Marker()
        if 'frame_id' in p:
            marker.header.frame_id = p['frame_id']
        else:
            marker.header.frame_id = "/camera_rgb_optical_frame"
        marker.ns = "alignment_markers"
        marker.id = len(markerArray.markers)
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        if 'scale' in p:
            marker.scale = p['scale']
        else:
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02
        if 'orientation' in p:
            marker.pose.orientation = p['orientation']
        else:
            marker.pose.orientation.x = 0.0;
            marker.pose.orientation.y = 0.0;
            marker.pose.orientation.z = 0.0;
            marker.pose.orientation.w = 1.0;
        if 'color' in p:
            marker.color.r = p['color'][0]
            marker.color.g = p['color'][1]
            marker.color.b = p['color'][2]
            marker.color.a = p['color'][3]
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        marker.pose.position = p['position']
        markerArray.markers.append(marker)
        
    while len(markerArray.markers) < 100:
        marker = Marker()
        marker.ns = "alignment_markers"
        marker.id = len(markerArray.markers)
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.header.frame_id = "/world"
        marker.scale.x = 0.0001
        marker.scale.y = 0.0001
        marker.scale.z = 0.0001
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.0
        marker.pose.position = Point(x=0,y=0,z=0)
        markerArray.markers.append(marker)
        
        
    return markerArray
    
def adjust_anchor(anchor):
    old_point = anchor['position']
    point = Point(x=old_point.x, y=old_point.y, z=old_point.z)
    anchor['position'] = point
    
    old_color = anchor['color']
    anchor['color'] = (0,1,0,1)
    
    static_pub.publish("/alignment_markers", create_marker_array(anchor))
    STEP = 0.01
    
    print("Move the green dot so that it is in the same position relative to the robot model as the actual marker on the physical")
    print("robot. Use the following keyboard to adjust the dot:")
    print("")
    print("    w/s - Up/Down")
    print("    a/d - Left/Right")
    print("    q/e - Forward/Backward")
    print("      1 - Set step size to 10cm")
    print("      2 - Set step size to 1cm")
    print("      3 - Set step size to 1mm")
    print("    Esc - Cancel")
    print("  Enter - Save")
    
    while True:
        c = baxter_external_devices.getch()
        if c:
            #catch Esc or ctrl-c
            if c in ['\x1b', '\x03']:
                anchor['position'] = old_point
                break
            elif c == 'w':
                point.y = point.y - STEP
            elif c == 's':
                point.y = point.y + STEP
            elif c == 'a':
                point.x = point.x - STEP
            elif c == 'd':
                point.x = point.x + STEP
            elif c == 'q':
                point.z = point.z + STEP
            elif c == 'e':
                point.z = point.z - STEP
            elif c == '1':
                STEP = 0.1
                print("Step set to 10cm.")
            elif c == '2':
                STEP = 0.01
                print("Step set to 1cm.")
            elif c == '3':
                STEP = 0.001
                print("Step set to 1mm.")
            elif c in ['\x0d']:
                break
    anchor['color'] = old_color
    
    
#The code for this function is from:
#http://nghiaho.com/?page_id=671
def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]; # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
       print "Reflection detected"
       Vt[2,:] *= -1
       R = Vt.T * U.T

    t = -R*centroid_A.T + centroid_B.T

    return R, t
    
def compute_transform(alignment_points):
    n = len(alignment_points)

    A = np.zeros((len(alignment_points), 3))
    for i, p in zip(range(len(alignment_points)), [x for _,x in alignment_points]):
        A[i,0] = p.x
        A[i,1] = p.y
        A[i,2] = p.z
    A = np.mat(A)
        
    B = np.zeros((len(alignment_points), 3))
    for i, p in zip(range(len(alignment_points)), [x for x,_ in alignment_points]):
        B[i,0] = p.x
        B[i,1] = p.y
        B[i,2] = p.z
    B = np.mat(B)

    # recover the transformation
    ret_R, ret_t = rigid_transform_3D(A, B)

    qw = math.sqrt(1.0 + ret_R[0,0] + ret_R[1,1] + ret_R[2,2])/2.0
    qx = (ret_R[2,1]-ret_R[1,2])/(4*qw)
    qy = (ret_R[0,2]-ret_R[2,0])/(4*qw)
    qz = (ret_R[1,0]-ret_R[0,1])/(4*qw)
    
    return np.reshape(ret_t, -1).tolist()[0] + [qx, qy, qz, qw]
    
def add_alignment_point(alignment_points):
    static_pub.publish("/alignment_markers", create_marker_array(*[dict(position=p, color=(0,0,1,1), frame_id='/camera_link') for _, p in alignment_points]))
    print("Please move the robot's arm so the green dot is clearly visible to the camera. Best results are obtained when the")
    print("correspondence points are spread out. Position the green dot on the robot's arm away from the blue points shown in")
    print("RViz.")
    point = None
    while True:
        if point is None:
            choice = cutil.select_from_list(["Find green dot in kinect frame"])
        else:
            choice = cutil.select_from_list(["Re-find green dot in kinect frame", "Save correspondence point"])
        
        if choice == 0:
            point, frame_id = find_the_green_dot()
            print("Red point indicates where the green dot was found.")
            static_pub.publish("/alignment_markers", create_marker_array(dict(position=point, color=(1,0,0,1)), *[dict(position=p, color=(0,0,1,1), frame_id='/camera_link') for _, p in alignment_points]))
        elif choice == 1:
            print("Which arm is the green dot on?")
            choice = cutil.select_from_list(["Left", "Right"], allow_cancel=False)
            if choice == 0:
                anchor = LEFT_ANCHOR
            else:
                anchor = RIGHT_ANCHOR
            alignment_points.append((cutil.transform_point(anchor['position'], anchor['frame_id'], '/world'), cutil.transform_point(point, frame_id, '/camera_link')))
            print("Added correspondence points: " + str(alignment_points[-1]))
            break
        else:
            break

def find_the_green_dot():
    def l2_norm(p1, p2):
        return (p1['x'] - p2['x'])*(p1['x'] - p2['x']) + (p1['y'] - p2['y'])*(p1['y'] - p2['y']) + (p1['z'] - p2['z'])*(p1['z'] - p2['z'])

    print("Finding the green dot...")
    print("\tGrabbing pointcloud...")
    data = cutil.ros_topic_get("camera/depth_registered/points", PointCloud2)
    cloud = pointclouds.pointcloud2_to_array(data, split_rgb=True)
    
    max_dist = 0.015*0.015
    colored_cloud = cloud[(cloud['b'] < 128) * (cloud['r'] < 120) * (cloud['g'] > 128)]
    print("\tGot " + str(len(colored_cloud)) + " green points. Constructing the graph...")
    #now lets construct a graph
    g = nx.Graph()
    for i1, p1 in enumerate(colored_cloud):
        for i2, p2 in enumerate(colored_cloud):
            if i2 <= i1:
                continue
            g.add_node(i1)
            g.add_node(i2)
            d = l2_norm(p1, p2)
            if d <= max_dist:
                g.add_edge(i1, i2)
    
    
    print("\tfinding cliques...")
    cliques = list(nx.find_cliques(g))
    print("\tfound cliques = " + str(len(cliques)))
    
    #now keep only nodes in the largest cliques
    max_clique = 0
    for c in cliques:
        if len(c) > max_clique:
            max_clique = len(c)
    
    popular_nodes = []
    num_max_cliques = 0
    for c in cliques:
        if len(c) < max_clique:
            continue
        num_max_cliques = num_max_cliques + 1
        for i in c:
            if i not in popular_nodes:
                popular_nodes.extend([i])
                
    for node in g.nodes():
        if node not in popular_nodes:
            g.remove_node(node)
          
    print("\tThe largest clique was " + str(max_clique) + ", and there were " + str(num_max_cliques) + " cliques of that size.")
        
    print("\tConstructing marker...")
    #now let's construct the marker array to output what we found    
    avgx = 0.0
    avgy = 0.0
    avgz = 0.0
    for node in g.nodes_iter():
        avgx += colored_cloud[node]['x']
        avgy += colored_cloud[node]['y']
        avgz += colored_cloud[node]['z']
        
    print("\tnumber of nodes = " + str(g.number_of_nodes()))
        
    avgx /= g.number_of_nodes()
    avgy /= g.number_of_nodes()
    avgz /= g.number_of_nodes()
    
    print("Found marker at (" + str(avgx) + ", " + str(avgy) + ", " + str(avgz) + ")")
    return Point(x=avgx, y=avgy, z=avgz), data.header.frame_id

def main():

    ros_init()
    trans = load_settings()
    align = [0]*len(trans)
    for i in range(len(trans)):
        align[i] = trans[i]
    
    def transform_pub_spin():
        r = rospy.Rate(100)
        br = tf.TransformBroadcaster()
        while True:
            br.sendTransform((align[0], align[1], align[2]), (align[3], align[4], align[5], align[6]), rospy.Time.now(), "/camera_link", "/world")
            r.sleep()
            
    thread = threading.Thread(target=transform_pub_spin, args=())
    thread.daemon = True
    thread.start()

    #print("Please find your \'kinect_frames.launch\' file (probably in something like \'/opt/ros/groovy/share/openni_launch\').")
    #print("If there is a node tag with the name=\"$(arg camera)_world\", then comment out the entire tag (using \'<!--\' to ")
    #print("start the comment and \'-->\' to end). You will probably need super-user permissions to do this.")
    print("If starting the kinect program from \'start_kinect.sh\', stop the program and run")
    print("\'roslaunch openni_launch openni.launch\'")
    print("instead. After alignment is done, either replace the first line of start_kinect.sh with the output from this program,")
    print("or make sure to run it before starting the kinect program (modifying start_kinect.sh is the recommended method).")
    print("")
    print("Press enter to continue...")
    cutil.getKeyboardLine()
    
    print("Please make sure you have RVIZ open and are displaying these three topics:")
    print("    -The model of the robot")
    print("    -The full point cloud visualization on /camera/depth_registered/points")
    print("    -The Marker Array topic /alignment_markers")
    print("")
    print("Press enter to continue...")
    cutil.getKeyboardLine()

    alignment_points = load_points()
    while True:
        static_pub.publish("/alignment_markers", create_marker_array(LEFT_ANCHOR, RIGHT_ANCHOR, *[dict(position=p, color=(0,0,1,1), frame_id='/camera_link') for _, p in alignment_points]))
        print("You currently have " + str(len(alignment_points)) + " correspondence points set.")
        print("Would you like to...")
        choice = cutil.select_from_list(["Add a point", "Delete a point", "Adjust anchor point on robot", "Compute transformation"])
        if choice == 0:
            add_alignment_point(alignment_points)
        elif choice == 1:
            print("Pick a point to delete or press Esc to cancel.")
            choice = cutil.select_from_list(["(% 7.4f, % 7.4f, % 7.4f) -> (% 7.4f, % 7.4f, % 7.4f)" % (cp.x, cp.y, cp.z, wp.x, wp.y, wp.z) for (cp, wp) in alignment_points])
            #yes, I did just do that :P
            if choice >= 0:
                cp, wp = alignment_points[choice]
                del alignment_points[choice]
                print("Removed (% 7.4f, % 7.4f, % 7.4f) -> (% 7.4f, % 7.4f, % 7.4f)" % (cp.x, cp.y, cp.z, wp.x, wp.y, wp.z))
            else:
                print("Canceled.")
        elif choice == 2:
            print("Which anchor would you like to adjust?")
            choice = cutil.select_from_list(["Left", "Right"])
            if choice == 0:
                adjust_anchor(LEFT_ANCHOR)
            elif choice == 1:
                adjust_anchor(RIGHT_ANCHOR)
        elif choice == 3:
            if len(alignment_points) < 3:
                print("You only have " + str(len(alignment_points)) + " points. You must have at least 3 to compute the transformation.")
            else:
                print("Computing transform...")
                trans = compute_transform(alignment_points)
                print("Got " + str(trans) + ".")
                print("")
                print("Replace the first line of \'start_kinect.sh\' with the following line in order to publish the transform between the kinect and baxter:")
                print("(alternatively, you may simply run the following line before starting the kinect program)")
                print("rosrun tf static_transform_publisher " + str(trans)[1:-1] + " world camera_link 100 &")
                print("")
                save_settings(trans)
                for i in range(len(trans)):
                    align[i] = trans[i]
        else:
            break
        save_points(alignment_points)

    static_pub.publish("/alignment_markers", None)
    rospy.signal_shutdown("Finished.")

  


def ros_init():
    print("Initializing node 'align_kinect_to_baxter'\n")
    rospy.init_node('align_kinect_to_baxter')


if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
