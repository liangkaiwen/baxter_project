#pragma once

#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>

template <typename KeypointPointT>
class Keypoints{
public:
	typedef KeypointPointT PointT;
	typedef pcl::PointCloud<KeypointPointT> CloudT;
		
public:
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	typename CloudT::Ptr keypoint_cloud;
	std::vector<int> inlier_count;


	// always allocate the cloud
	Keypoints() :
		keypoint_cloud(new CloudT)
	{
	}

	Keypoints(const Keypoints<KeypointPointT>& other) :
		keypoint_cloud(new CloudT)
	{
		*this = other;
	}

	void operator=(const Keypoints<KeypointPointT>& other) {
		keypoints = other.keypoints;
		descriptors = other.descriptors.clone();
		keypoint_cloud.reset(new CloudT); // don't need this
		pcl::copyPointCloud(*other.keypoint_cloud, *keypoint_cloud);
		inlier_count = other.inlier_count;
	}

	void filter(const std::vector<bool>& keep_keypoints) {
		if (keep_keypoints.size() != keypoints.size()) throw std::runtime_error ("keep_keypoints wrong size");
		if (keypoints.empty()) return; // so we are now sure that we have at least one keypoint
		// keep the original values
		std::vector<cv::KeyPoint> keypoints_old = keypoints;
		cv::Mat descriptors_old = descriptors;
		typename CloudT::Ptr keypoint_cloud_old = keypoint_cloud;
		std::vector<int> inlier_count_old = inlier_count;
		// reset the internal storage
		keypoints.clear();
		descriptors = cv::Mat(0, descriptors_old.cols, descriptors_old.type());
		keypoint_cloud.reset(new CloudT);
		inlier_count.clear();
		// add in the keypoints that are true
		for (size_t i = 0; i < keep_keypoints.size(); i++) {
			if (keep_keypoints[i]) {
				keypoints.push_back(keypoints_old[i]);
				descriptors.push_back(descriptors_old.row(i));
				keypoint_cloud->points.push_back(keypoint_cloud_old->points[i]);
				inlier_count.push_back(inlier_count_old[i]);
			}
		}
		updateCloudDims();
	}

	void append(const Keypoints<KeypointPointT>& other) {
		for (size_t i = 0; i < other.keypoints.size(); i++) {
			keypoints.push_back(other.keypoints[i]);
			descriptors.push_back(other.descriptors.row(i));
			keypoint_cloud->points.push_back(other.keypoint_cloud->points[i]);
			inlier_count.push_back(other.inlier_count[i]);
		}
		updateCloudDims();
	}

	void updateCloudDims()
	{
		keypoint_cloud->width = keypoint_cloud->size();
		keypoint_cloud->height = 1;
		keypoint_cloud->is_dense = true;
	}

	size_t size() const {
		return keypoints.size();
	}
};
