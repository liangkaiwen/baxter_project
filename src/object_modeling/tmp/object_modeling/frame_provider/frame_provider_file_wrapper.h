#pragma once

#include "frame_provider_base.h"

class FrameProviderFileWrapper : public FrameProviderBase
{
public:
    FrameProviderFileWrapper(boost::shared_ptr<FrameProviderBase> frame_provider_ptr, int frame_increment = 1, int frame_start = -1, int frame_end = -1);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	virtual bool getLastFramePose(Eigen::Affine3f & camera_pose_result);

	virtual bool getLastROSTimestamp(ROSTimestamp & ros_timestamp_result);

    virtual bool getLastFilename(std::string & result_filename);


protected:
	// members
	int frame_start;
	int frame_end;
	int frame_increment;

    int frame_counter_actual; // incorporates the frames skipped (so +1 for ACTUAL frames returned, not skipped frames)

	boost::shared_ptr<FrameProviderBase> frame_provider_ptr;
};
