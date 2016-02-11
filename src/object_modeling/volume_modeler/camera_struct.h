#pragma once

#include "ros_timestamp.h"

struct CameraStruct {
	Eigen::Affine3f pose;
	Eigen::Affine3f original_pose; // first setting of pose
	ROSTimestamp ros_timestamp;

	CameraStruct(Eigen::Affine3f const& pose, ROSTimestamp const& ros_timestamp)
		: pose(pose),
		original_pose(pose),
		ros_timestamp(ros_timestamp)
	{
	}


	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};
typedef boost::shared_ptr<CameraStruct> CameraStructPtr;