#pragma once

#include "frame_provider_base.h"


class FrameProviderPNGDepth : public FrameProviderBase
{
public:
    FrameProviderPNGDepth(fs::path folder, float depth_factor = 10000.f, fs::path camera_list_txt = fs::path(), cv::Scalar fake_color = cv::Scalar::all(128));

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	virtual bool getLastFramePose(Eigen::Affine3f & camera_pose_result);

protected:
	// members
	float depth_factor_;
	std::vector<fs::path> files_;
	std::vector<fs::path>::iterator file_iter_;

    // a bit hacky
    Eigen::Affine3f last_frame_pose_;
	std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > camera_list_vec_;

	// fake color?
	cv::Scalar fake_color_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

bool is_not_png_depth_image(fs::path filename);
