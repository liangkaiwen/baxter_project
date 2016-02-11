#pragma once

#include "frame_provider_base.h"

#include "params_camera.h"

class FrameProviderFreiburg : public FrameProviderBase
{
public:
	FrameProviderFreiburg(fs::path associate_file);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	virtual bool getLastROSTimestamp(ROSTimestamp & ros_timestamp_result);

protected:
	// types
	struct FrameInformation {
		ROSTimestamp rgb_timestamp;
		fs::path rgb_path;
		ROSTimestamp depth_timestamp;
		fs::path depth_path;
	};


	// members
	std::vector<FrameInformation> frame_list_;
	std::vector<FrameInformation>::iterator frame_list_iterator_;

	// set this so subsequent calls blah blah
	ROSTimestamp last_timestamp_;
};
