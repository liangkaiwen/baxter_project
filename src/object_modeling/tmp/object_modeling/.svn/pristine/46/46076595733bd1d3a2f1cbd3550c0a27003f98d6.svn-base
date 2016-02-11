#pragma once

#include "frame_provider_base.h"

#include "EigenUtilities.h"

class FrameProviderLuis : public FrameProviderBase
{
public:
    FrameProviderLuis(fs::path folder, float depth_factor = 1000.f, fs::path luis_object_pose_txt = fs::path());

	virtual bool getNextFrame(cv::Mat & color, cv::Mat & depth);

    virtual void skipNextFrame();

	virtual void reset();

	virtual bool getLastFramePose(Eigen::Affine3f & camera_pose_result);

protected:
	// members
	float depth_factor_;
	std::vector<fs::path> files_;
	std::vector<fs::path>::iterator file_iter_;

	// a big of a hack
    PosePtrList camera_list_vec_;
	Eigen::Affine3f last_frame_pose_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

bool is_not_luis_color_image(fs::path filename);
