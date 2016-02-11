"""
2/11/16
Kevin Liang
UW RSE Lab

This class helps publish markers for visualizing data in rviz
"""

import pcl
import rospy

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from geometry_msgs.msg import (
	PoseStamped,
	Pose,
	Point,
	Quaternion,
	QuaternionStamped,
	PointStamped
)

class Marker_Pub(object):
	def __init__(self):
		rospy.init_node("marker_pub")
		self._rate = 5.0 #Hz
		self._publisher = rospy.Publisher("/marker_pub", MarkerArray, queue_size = 10)
		self._marker_array = None

	# sets the internal markers to an array of point type markers
	# where points is an array of 3-dimensional coordinates, in the order of x,y,z
	# frame of reference is /world by default
	# setting log to true will print logging information to the console
	def set_points(points, frame="/world", log=False):
		self._marker_array = MarkerArray()
		count = 0
		for point in points:
			marker = Marker()
			marker.header.frame_id = frame
			marker.header.stamp = rospy.Time(0) #rospy.Time.now()
			marker.type = marker.SPHERE
			marker.action = marker.ADD
			marker.pose.position.x = point[0]
			marker.pose.position.y = point[1]
			marker.pose.position.z = point[2]
			marker.scale.x = 0.01
			marker.scale.y = 0.01
			marker.scale.z = 0.01
			marker.color.a = 1.0
			marker.color.r = 1.0
			marker.color.g = 0.0
			marker.color.b = 0.0
			marker.id = count
			count += 1

			self._marker_array.markers.append(marker)
			if log:
				print "Adding point - x:" + str(point[0]) +
				                    " y:" + str(point[1]) +
				                    " z:" + str(point[2])
	    if log:
	    	print "Total marker count: " + str(count)

	# sets the internal markers to an array of vector type markers
	# where vectors are represented as a point and 3-dimensional direction
	# vector_points is expected to be an array of origin points for the vectors
	# and vector_dir is expected to be an array with direction for the corresponding point
	def set_vectors(vector_points, vector_dir, frame="/world", log=False):
		self._marker_array = MarkerArray()
		count = 0
		if (len(vector_points) != len(vector_dir)): 		# input sanity check
			print "Invalid vectors - no direction for every point"
			return
		for indx in range(len(vector_points)):
			unit_vector = [None] * 3
			for j in range(3):
				unit_vector[i] = vector_points[indx][i] + vector_dir[indx][i]

			vector_marker = marker()
			vector_marker.header.frame_id = frame
			vector_marker.header.stamp = rospy.Time(0)
			vector_marker.type = vector_marker.ARROW
			vector_marker.action = vector_marker.ADD
			vector_marker.points = [create_point_tuple(vector_points[indx]),
			                        create_point_tuple(unit_vector)]
			vector_marker.color.a = 1.0
			vector_marker.color.r = 0.0
			vector_marker.color.g = 1.0
			vector_marker.color.b = 0.0
			vector_marker.scale.x = 0.01
			vector_marker.scale.y = 0.02
			vector_marker.id = count
			count += 1

			self._marker_array.markers.append(vector_marker)

			if log:
				print "Added vector -  x:" + str(vector_points[indx][0]) +
				                     " y:" + str(vector_points[indx][1]) +
				                     " z:" + str(vector_points[indx][2])
				print "dx:" + str(vector_dir[indx][0]) +
				     " dy:" + str(vector_dir[indx][1]) +
				     " dz:" + str(vector_dir[indx][2])

		if log:
			print "Total vector count: " + str(count)

	def set_lines(points, frame="/world", log=False):
		self._marker_array = MarkerArray()

		lines = Marker()
		lines.header.frame_id = frame
		lines.header.stamp = rospy.Time(0)
		lines.type = marker.LINE_LIST
		lines.action = marker.ADD
		lines.scale.x = 0.02
		lines.color.a = 0.0
		lines.color.r = 0.0
		lines.color.b = 1.0
		lines.points = points

		marker_array.markers.append(lines)

		if log:
			print "Total line count: " + str(len(points) / 2)

	# publishes the current stored marker array data
	# returns true if successful, else false
	def publish():
		if self._marker_array:
			self._publisher.publish(self._marker_array)
			return True
		return False

	# returns a Point at coordinate x,y,z
	def create_point(x, y, z):
		point = Point()
		point.x = x
		point.y = y
		point.z = z
		return point

	# creates a Point from a 3-tuple
	def create_point_tuple(pt_tuple):
		return create_point(pt_tuple[0],
			                pt_tuple[1],
			                pt_tuple[2])

# doesn't quite do anything yet
# default behavior of class yet to be determined
if __name__ == "__main__":
	marker_pub = Marker_Pub()
	rate = rospy.Rate(marker_pub._rate)
	while not rospy.is_shutdown():
		marker_pub._publisher.publish(self._marker_array)
		rate.sleep()