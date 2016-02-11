#pragma once

#include "frame_provider_base.h"

/*
Current format expected:
#.jpg
#.png
*/
class FrameProviderYuyin : public FrameProviderBase
{
public:
	FrameProviderYuyin(fs::path folder, float depth_factor, std::string required_prefix);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

protected:
	// members
	float depth_factor_;
	std::vector<fs::path> files_;
	std::vector<fs::path>::iterator file_iter_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

bool is_not_jpg_image(fs::path filename);
bool is_not_png_image(fs::path filename);
