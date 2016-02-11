import rospy
from std_msgs.msg import String

class Camera_Publisher(object):
	def __init__(self):
		rospy.init_node("baxter_hand_cam")
		self._rate = 30.0 #Hz

	def publish():
		pub = rospy.Publisher('chatter', String, queue_size=10)
		rate = rospy.Rate(self._rate)
		while not rospy.is_shutdown():
			pub.publish(data)
			rate.sleep()

if __name__ == '__main__':
	camera_pub = Camera_Publisher()
	try:
		camera_pub.publish()
	except rospy.ROSInterruptException:
		pass
