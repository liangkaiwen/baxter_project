#pragma once

#include "basic.h"
#include "frame.h"
#include "keypoints.h"
#include "params_camera.h"
#include "params_features.h"

class FeatureMatching
{
public:
	FeatureMatching(ParamsCamera const& params_camera, ParamsFeatures const& params_features);

	void computeFeatures(const Frame & frame, Keypoints & keypoints);

	void addFeaturesForFrameIfNeeded(Frame & frame);

	void matchDescriptors(const Keypoints & keypoints_source, const Keypoints & keypoints_target, std::vector< cv::DMatch > & matches);

	// DON'T CALL THIS ONE YET
	bool ransacOpenCV(const Keypoints & keypoints_source, const Keypoints & keypoints_target, const std::vector<cv::DMatch> & matches, Eigen::Affine3f & pose, std::vector<cv::DMatch> & inliers);

	bool ransac(const Keypoints & keypoints_source, const Keypoints & keypoints_target, const std::vector<cv::DMatch> & matches, Eigen::Affine3f & pose, std::vector<cv::DMatch> & inliers);

	void setVerbose(bool value) {verbose = value;}

protected:
	ParamsCamera const& params_camera;
	ParamsFeatures const& params_features;

	cv::Ptr<cv::FeatureDetector> detector_ptr;
	cv::Ptr<cv::DescriptorExtractor> extractor_ptr;
	cv::Ptr<cv::DescriptorMatcher> matcher_ptr;	

	bool verbose;
};
