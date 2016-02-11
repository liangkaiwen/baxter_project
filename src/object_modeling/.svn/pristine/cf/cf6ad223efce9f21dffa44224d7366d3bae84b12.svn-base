#pragma once

#include "basic.h"

struct Keypoints
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	Eigen::Matrix4Xf points;

	void write(cv::FileStorage & fs) const;
	void read(const cv::FileNode & node);
};

namespace cv {
void write(cv::FileStorage& fs, const std::string&, const Keypoints& x);
void read(const cv::FileNode& node, Keypoints& x, const Keypoints& default_value = Keypoints());
}
