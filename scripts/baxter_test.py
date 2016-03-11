import rospy
import tf
import baxter_interface
import baxter_tools
import baxter_examples
import pcl
import numpy
import pointclouds

#Testing imports
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sklearn.decomposition import PCA
import math

from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from baxter_interface import CHECK_VERSION
from std_msgs.msg import Header
from geometry_msgs.msg import (
	PoseStamped,
	Pose,
	Point,
	Quaternion,
	QuaternionStamped,
	PointStamped
)
from baxter_core_msgs.srv import (
	SolvePositionIK,
	SolvePositionIKRequest,
)
import struct
import sys

class Baxter_Test(object):
	def __init__(self):
		rospy.init_node("baxter_test")
		self._rs = baxter_interface.RobotEnable(CHECK_VERSION)
		self._init_state = self._rs.state().enabled
		self._rs.enable()
		self._rate = 5.0 #Hz

		#IK
		self.ns = "ExternalTools/left/PositionKinematicsNode/IKService"
		self.hdr = Header(stamp = rospy.Time(0), frame_id = "base")
		self.iksvc = rospy.ServiceProxy(self.ns, SolvePositionIK)
		self.ikreq = SolvePositionIKRequest()
		self._left_arm = baxter_interface.limb.Limb("left")
		self._right_arm = baxter_interface.limb.Limb("right")
		self._left_joint_names = self._left_arm.joint_names()
		self._right_joint_names = self._right_arm.joint_names()
		self._left_angles = [self._left_arm.joint_angle(joint) for joint in self._left_joint_names]
		self._right_angles = [self._right_arm.joint_angle(joint) for joint in self._right_joint_names]

	def update_angles(self):
		self._left_angles = [self._left_arm.joint_angle(joint) for joint in self._left_joint_names]
		self._right_angles = [self._right_arm.joint_angle(joint) for joint in self._right_joint_names]

	def move_left_to_coord(self, point):
		point_above = Point(point[0], point[1], point[2] + 0.2)
		print point_above
		pose = PoseStamped(
			header = self.hdr,
			pose = Pose(
				position = point_above,
				orientation = Quaternion(
					x=-0.366894936773,
                    y=0.885980397775,
                    z=0.108155782462,
                    w=0.262162481772,
				),
			),
		)
		self.ikreq.pose_stamp.append(pose)
		try:
			rospy.wait_for_service(self.ns, 5.0)
			resp = self.iksvc(self.ikreq)
		except (rospy.ServiceException, rospy.ROSException), e:
			rospy.logerr("Service call failed: %s" % (e,))
			return 1
		resp_seeds = struct.unpack('<%dB' % len(resp.result_type), 
									resp.result_type)
		print resp_seeds
		if (resp_seeds[0] != resp.RESULT_INVALID):
			joint_solution = dict(zip(resp.joints[0].name, resp.joints[0].position))
			print joint_solution
			self._left_arm.move_to_joint_positions(joint_solution)
			print "done"
		else:
			print "Invalid POSE - No Valid Joint Solution Found"
		return 0

		
def main():
	baxter_test = Baxter_Test()
	rate = rospy.Rate(baxter_test._rate)
	#retrieve point cloud data from Baxter
	point_cloud_msg = ros_topic_get("/camera/depth_registered/points", PointCloud2)
	point_cloud_array = pointclouds.pointcloud2_to_xyz_array(point_cloud_msg)
	point_cloud_array = transform_matrix(point_cloud_array, "/camera_rgb_optical_frame", "/base")
	point_cloud = pcl.PointCloud() #create new pcl PointCloud
	point_cloud.from_list(point_cloud_array) #import data to PointCloud from 2d matrix

	#Experimental code for the hand mounted camera
	print "looking up hand camera"
	"""f200_cloud_msg = ros_topic_get("/camera/depth/points", PointCloud2)
	print f200_cloud_msg
	point_cloud_array = pointclouds.pointcloud2_to_xyz_array(f200_cloud_msg)
	print point_cloud_array"""

	#filter data
	fil = point_cloud.make_passthrough_filter()
	fil.set_filter_field_name("x")
	fil.set_filter_limits(0.5, 1)
	cloud_filtered_x = fil.filter()

	fil = cloud_filtered_x.make_passthrough_filter()
	fil.set_filter_field_name("z")
	fil.set_filter_limits(-0.1, 0.5)
	cloud_filtered = fil.filter()

	#segment data
	seg = cloud_filtered.make_segmenter()
	seg.set_model_type(pcl.SACMODEL_PLANE)
	seg.set_method_type(pcl.SAC_RANSAC)
	seg.set_distance_threshold(0.03)
	indices, model = seg.segment()

	#remove table plane based on model
	object_cloud = filter_plane(model, cloud_filtered, 0.1)
	print object_cloud

	X = object_cloud.to_array()
	#our PCA only needs to deal with x-y plane, remove z values
	X = X[:,[0,1]]
	pca = PCA(n_components=2)
	X = pca.fit_transform(X)

	#calculate object centroid
	object_centroid = get_centroid(object_cloud)

	#calculate object bounding box
	bounding_box = fit_bounding_box(object_cloud, pca.components_, object_centroid)
	#baxter_test.move_left_to_coord(object_centroid)

	
	publisher = rospy.Publisher("/baxter_test", MarkerArray, queue_size = 10)
	marker_array = MarkerArray()

	
	#PCA vector sanity check (if needed)
	"""count = 0
	for vector in pca.components_:
		vector_marker = Marker()
		vector_marker.header.frame_id = "/base"
		vector_marker.header.stamp = rospy.Time(0)
		vector_marker.type = vector_marker.ARROW
		vector_marker.action = vector_marker.ADD
		start = Point()
		vector_marker.points = [createPoint(object_centroid[0], object_centroid[1], 0),
								createPoint(object_centroid[0] + vector[0], 
								object_centroid[1] + vector[1], 0)] 
		if count == 0:
			vector_marker.color.a = 1.0 #first vector green
			vector_marker.color.r = 0.0
			vector_marker.color.g = 1.0
			vector_marker.color.b = 0.0
		else:
			vector_marker.color.a = 1.0 #second vector blue
			vector_marker.color.r = 0.0
			vector_marker.color.g = 0.0
			vector_marker.color.b = 1.0
		vector_marker.scale.x = 0.01
		vector_marker.scale.y = 0.02
		vector_marker.id = count
		marker_array.markers.append(vector_marker)
		count += 1"""

	#continually do stuff
	while not rospy.is_shutdown():
		publisher.publish(marker_array)
		#baxter_test.move_left_to_coord(object_centroid)
		rate.sleep()


#==============Utility Methods========================================================


#model is a plane represented as a list of coefficients [a, b, c, d] 
#where the plane is ax + by + cz + d = 0
#points_list is a list of 3 dimensional tuples
#tolerance is a float value of how "far" from the plane a point must be to be accepted
def filter_plane(model, point_cloud, tolerance):
	points_list = point_cloud.to_list()
	filtered_list = []
	a = model[0]
	b = model[1]
	c = model[2]
	d = model[3]
	for point in points_list:
		x = point[0]
		y = point[1]
		z = point[2]
		val = a * x + b * y + c * z + d
		if (abs(val) > tolerance): #point falls outside of the plane
			filtered_list.append(point)
	filtered_cloud = pcl.PointCloud()
	filtered_cloud.from_list(filtered_list)
	return filtered_cloud

#point_cloud is a PCL point cloud object
#returns a tuple (x, y, z) which is the 3d coordinate of the centroid of the cloud
def get_centroid(point_cloud):
	numPoints = point_cloud.size
	if numPoints == 0:
		return None
	points_list = point_cloud.to_list()
	x = 0.0
	y = 0.0
	z = 0.0
	for point in points_list:
		x += point[0]
		y += point[1]
		z += point[2]
	return  (x/numPoints, y/numPoints, z/numPoints)

def ros_topic_get(topic, msg_type, timeout=-1):
	val = [None]
	def callback(msg):
		val[0] = msg
	sub = rospy.Subscriber(topic, msg_type, callback)

	t = rospy.Time.now()
	while (timeout <= 0 or (rospy.Time.now() - t).to_sec() <= timeout) and (val[0] is None):
			rospy.sleep(0.01)
	sub.unregister()
	return val[0]

transform_point_listener = None  
#rosrun tf tf_monitor
#rosrun tf view_frames
def transform_point(point, from_frame, to_frame, verbose=False):
    if from_frame == to_frame:
        return point
        
    return_list = True
    if not isinstance(point, list):
        point = [point]
        return_list = False

    mat = numpy.zeros((len(point), 3))
    for i in range(len(point)):
        if isinstance(point[i], list) or isinstance(point[i], tuple):
            mat[i,0] = point[i][0]
            mat[i,1] = point[i][1]
            mat[i,2] = point[i][2]
        else:
            mat[i,0] = point[i].x
            mat[i,1] = point[i].y
            mat[i,2] = point[i].z
    
    mat = transform_matrix(mat, from_frame, to_frame, verbose=verbose)
    ret = []
    for row in mat:
        ret.append(Point(x=row[0], y=row[1], z=row[2]))
    
    if return_list:
        return ret
    else:
        return ret[0]
        
def transform_matrix(mat, from_frame, to_frame, verbose=False):
    if from_frame is to_frame:
        return mat
        
    global transform_point_listener
    if transform_point_listener is None:
        transform_point_listener = tf.TransformListener()
    rate = rospy.Rate(1.0)
    
    if verbose:
        print("Looking up transform from " + from_frame + " to " + to_frame + " for " + str(mat.shape) + " matrix.") 
    
    while True:
        try:
            hdr = Header(stamp=rospy.Time(0), frame_id=from_frame)
            mat44 = transform_point_listener.asMatrix(to_frame, hdr)
        except(tf.LookupException):
            if verbose:
                print("lookup excpetion")
            rate.sleep()
            continue
        except(tf.ExtrapolationException):
            if verbose:
                print("extrapolation excpetion")
            rate.sleep()
            continue
        break
    
    a = numpy.empty((mat.shape[0], 1))
    a.fill(1.0)
    a = numpy.hstack((mat, a))
    a = numpy.transpose(numpy.dot(mat44, numpy.transpose(a)))
    return numpy.delete(a, -1, 1)
        
# cloud must be a numpy recordarray.
def transform_cloud(cloud, from_frame, to_frame):
        if from_frame == to_frame:
            return cloud
            
        #filter out all x, y, or z's that are NaN
        mask = numpy.isfinite(cloud['x']) & numpy.isfinite(cloud['y']) & numpy.isfinite(cloud['z'])
        cloud = cloud[mask]
        
        cloud_array = pointclouds.get_xyz_points(cloud)
        cloud_array = transform_matrix(cloud_array, from_frame, to_frame, verbose=False)
        return pointclouds.xyz_to_cloud(cloud, cloud_array)

# returns a Point at coordinate x,y,z
def createPoint(x, y, z):
	point = Point()
	point.x = x
	point.y = y
	point.z = z
	return point

def createPoint_tuple(pt_tuple):
	point = Point()
	point.x = pt_tuple[0]
	point.y = pt_tuple[1]
	point.z = pt_tuple[2]
	return point


#returns an array of 3d point tuples indicating the corners of the best fit bounding box
#takes a PointCloud object_cloud and
#vectors where are 2d vectors [x, y] as the ratio of x to y
def fit_bounding_box(object_cloud, pca_vectors, centroid):
	#output array (8 corners of the bounding box)
	# 0 - MaxD1, MaxD2, MaxZ
	# 1 - MinD1, MaxD2, MaxZ
	# 2 - MaxD1, MinD2, MaxZ
	# 3 - MinD1, MinD2, MaxZ
	# 4 - MaxD1, MaxD2, MinZ
	# 5 - MinD1, MaxD2, MinZ
	# 6 - MaxD1, MinD2, MinZ
	# 7 - MinD1, MinD2, MinZ
	output = []
	first_pt = True
	maxD1 = None
	minD2 = None
	maxZ = None

	minD1 = None
	minD2 = None
	minZ = None

	d1 = pca_vectors[0]
	d2 = pca_vectors[1]

	# shift up in the direction of the vector by 1 unit 
	# from the centroid (origin in frame of PCA)
	pt_d1 = [centroid[0] + d1[0], centroid[1] + d1[1]] 
	pt_d2 = [centroid[0] + d2[0], centroid[1] + d2[1]]

	slope_d1 = (pt_d1[1] - centroid[1]) / (pt_d1[0] - centroid[0])
	slope_d2 = (pt_d2[1] - centroid[1]) / (pt_d2[0] - centroid[0])

	for point in object_cloud:
		d1_val = distance_to_line(point, centroid, d1)
		d2_val = distance_to_line(point, centroid, d2)
		slope_pt = (point[1] - centroid[1]) / (point[0] - centroid[0])

		# determine sign of point dimensions along new axes
		if slope_d1 > slope_pt and slope_d2 < slope_pt:
			#d1 positive
			#d2 positive
			d1_val = d1_val * 1
		elif slope_d1 > slope_pt and slope_d2 > slope_pt:
			#d1 negative
			d1_val = d1_val * -1
			#d2 positive
		elif slope_d1 < slope_pt and slope_d2 < slope_pt:
			#d1 negative
			d1_val = d1_val * -1
			#d2 negative
			d2_val = d2_val * -1
		elif slope_d1 < slope_pt and slope_d2 > slope_pt:
			#d1 positive
			#d2 negative
			d2_val = d2_val * -1

		#determine if point is an extreme
		if first_pt:
			maxD1 = d1_val
			maxD2 = d2_val
			minD1 = d1_val
			minD2 = d2_val
			maxZ = point[2]
			minZ = point[2]
			first_pt = False
		else:
			if d1_val > maxD1:
				maxD1 = d1_val
			elif d1_val < minD1:
				minD1 = d1_val
			if d2_val > maxD2:
				maxD2 = d2_val
			elif d2_val < minD2:
				minD2 = d2_val

			if point[2] > maxZ:
				maxZ = point[2]
			elif point[2] < minZ:
				minZ = point[2]

	d1_extremes = [maxD1, minD1]
	d2_extremes = [maxD2, minD2]
	z_extremes = [maxZ, minZ]

	#permute the corner points (in frame of d1, d2)
	for i in range(2):
		for j in range(2):
			for k in range(2):
				output.append([d1_extremes[k],
					           d2_extremes[j],
					           z_extremes[i]])

	#transform points back into x-y frame
	for indx in range(len(output)):
		output[indx] = transform_frame(output[indx], d1, d2, centroid)

	return output


# returns absolute min distance to line
# http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
def distance_to_line(point, line_pt1, line_pt2):
	return (math.fabs((line_pt2[0] - line_pt1[0]) * (point[1] - line_pt1[1]) - 
		   (line_pt1[0] - point[0]) * (line_pt2[1] - line_pt1[1])) / 
	       (math.sqrt(math.pow(line_pt2[0] - line_pt1[0], 2) + math.pow(line_pt2[1] - line_pt1[1], 2))))


# takes a 3 dimensional point and transforms the first 2 dimensions
# from the frame (given by axes d1, d2, and center point (all with values in the x-y frame) 
# to the standard x, y, z cartesian coordinate frame
def transform_frame(point, d1_vector, d2_vector, center):
	print point
	d1_theta = math.atan2(d1_vector[1], d1_vector[0])
	d2_theta = math.atan2(d2_vector[1], d2_vector[0])

	d1_x_comp = point[0] * math.cos(d1_theta)
	d1_y_comp = point[0] * math.sin(d1_theta)

	d2_x_comp = point[1] * math.cos(d2_theta)
	d2_y_comp = point[1] * math.sin(d1_theta)

	return [center[0] + d1_x_comp + d2_x_comp,
	        center[1] + d1_y_comp + d2_y_comp,
	        point[2]]



"""executes the script if this is the main file"""
if __name__ == '__main__':
	main()