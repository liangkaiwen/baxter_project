#include "frame_provider_base.h"

FrameProviderBase::FrameProviderBase()
{
}

bool FrameProviderBase::getLastFramePose(Eigen::Affine3f & camera_pose_result)
{
    return false;
}

bool FrameProviderBase::getLastROSTimestamp(ROSTimestamp & ros_timestamp_result)
{
	return false;
}

bool FrameProviderBase::getLastFilename(std::string & result_filename)
{
    return false;
}
