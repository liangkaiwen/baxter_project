#pragma once

// opencl lib
#include <ImageBuffer.h>

// pcl
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template <typename PointT, typename KeypointPointT>
class Frame{
public:
	//typedef PointT PointT;
	typedef pcl::PointCloud<PointT> CloudT;
	//typedef typename Keypoints<KeypointPointT>::PointT KeypointPointT;
	typedef typename Keypoints<KeypointPointT>::CloudT KeypointCloudT;

	// it all comes from this:
	typename CloudT::Ptr cloud_ptr;

	// the rest of these come from addImagesToFrame, masking, and keypoints
	cv::Mat image_color;
	cv::Mat image_color_hsv;
	cv::Mat image_depth;
	cv::Mat depth_mask;
	cv::Mat depth_mask_without_hand;
	cv::Mat object_mask; // frame size
	cv::Mat object_cloud_normal_mask; // object_cloud_ptr size
	cv::Rect object_rect;
	Keypoints<KeypointPointT> object_kp;
	typename CloudT::Ptr object_cloud_ptr;
	typename pcl::PointCloud<pcl::Normal>::Ptr object_normal_cloud_ptr;
	typename KeypointCloudT::Ptr object_kp_projection_cloud_ptr;
	
	boost::shared_ptr<ImageBuffer> image_buffer_points_ptr;
	boost::shared_ptr<ImageBuffer> image_buffer_normals_ptr;

	void addImagesToFrame() {
		unsigned int rows = cloud_ptr->height;
		unsigned int cols = cloud_ptr->width;

		image_color = cv::Mat(rows, cols, CV_8UC3);
		image_depth = cv::Mat(rows, cols, CV_32FC1);
		depth_mask = cv::Mat(rows, cols, CV_8UC1);

		for (unsigned int row = 0; row < rows; row++) {
			for (unsigned int col = 0; col < cols; col++) {
				PointT& p = cloud_ptr->at(col, row);
				// color
				image_color.at<cv::Vec3b>(row, col)[0] = p.b;
				image_color.at<cv::Vec3b>(row, col)[1] = p.g;
				image_color.at<cv::Vec3b>(row, col)[2] = p.r;

				// depth_image
				image_depth.at<float>(row, col) = p.z;

				// valid?
				depth_mask.at<uchar>(row, col) = p.z > 0 ? 255 : 0;
			}
		}

		// Make an HSV version (masking)
		cv::cvtColor(image_color, image_color_hsv, CV_BGR2HSV);
	}
};
