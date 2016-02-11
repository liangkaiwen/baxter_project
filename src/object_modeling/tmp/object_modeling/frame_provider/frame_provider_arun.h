#pragma once

#include "frame_provider_base.h"

/*
Current format expected:
#.jpg
#.png
*/
class FrameProviderArun : public FrameProviderBase
{
public:
	FrameProviderArun(fs::path folder, float depth_factor);

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

    virtual bool getLastFilename(std::string &result_filename);


    static bool is_not_png_image(fs::path filename);

protected:
	// members
	float depth_factor_;
	std::vector<fs::path> files_;
	std::vector<fs::path>::iterator file_iter_;

    fs::path last_color_file_path_;


public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};


