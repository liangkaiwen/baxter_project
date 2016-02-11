#pragma once

#include "frame_provider_base.h"

/*
Current format expected:
#####-color.png
#####-depth.png
*/
class FrameProviderPNG : public FrameProviderBase
{
public:
	FrameProviderPNG(fs::path folder, float depth_factor = 10000.f, bool expect_pose = false, fs::path camera_list_txt = fs::path());

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	virtual bool getLastFramePose(Eigen::Affine3f & camera_pose_result);

protected:
	// members
	float depth_factor_;
	std::vector<fs::path> files_;
	std::vector<fs::path>::iterator file_iter_;

	// this is a bit hacked in here:
	bool expect_pose_;
	Eigen::Affine3f last_frame_pose_;

	// so is this (for loading a single file of poses)
	std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > camera_list_vec_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

bool is_not_png_color_image(fs::path filename);
