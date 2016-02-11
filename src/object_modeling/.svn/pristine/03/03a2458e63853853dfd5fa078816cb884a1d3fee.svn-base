#pragma once

#include "parameters.h" // for params for image to cloud

template<typename PointT>
cv::Mat cloudToImage(const pcl::PointCloud<PointT>& cloud, cv::Vec3b nanColor = cv::Vec3b(0,0,0))
{
	unsigned int rows = cloud.height;
	unsigned int cols = cloud.width;

	cv::Mat image_color(rows, cols, CV_8UC3);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const PointT& p = cloud.at(col, row);

			if (pcl_isnan(p.z)) {
				image_color.at<cv::Vec3b>(row, col) = nanColor;
			}
			else {
				image_color.at<cv::Vec3b>(row, col)[0] = p.b;
				image_color.at<cv::Vec3b>(row, col)[1] = p.g;
				image_color.at<cv::Vec3b>(row, col)[2] = p.r;
			}
		}
	}

	return image_color;
}

template<typename PointT>
cv::Mat cloudToColorDepthImage(const pcl::PointCloud<PointT>& cloud, float min_depth, float max_depth, cv::Vec3b nanColor = cv::Vec3b(0,0,255))
{
	unsigned int rows = cloud.height;
	unsigned int cols = cloud.width;

	cv::Mat image_color(rows, cols, CV_8UC3);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const PointT& p = cloud.at(col, row);

			if (pcl_isnan(p.z)) {
				image_color.at<cv::Vec3b>(row, col) = nanColor;
			}
			else {
				float normalized = 0;
				if (p.z < min_depth) {
					normalized = 1;
				}
				else if (p.z > max_depth) {
					normalized = 0;
				}
				else {
					normalized = 1 - (p.z - min_depth) / (max_depth - min_depth);
				}

				image_color.at<cv::Vec3b>(row, col)[0] = 255 * normalized;
				image_color.at<cv::Vec3b>(row, col)[1] = 255 * normalized;
				image_color.at<cv::Vec3b>(row, col)[2] = 255 * normalized;
			}
		}
	}

	return image_color;
}

template<typename PointT>
cv::Mat cloudToColorDepthImage(const pcl::PointCloud<PointT>& cloud, cv::Vec3b nanColor = cv::Vec3b(0,0,255))
{
	// compute min and max, then call the other one
	float min_depth = FLT_MAX;
	float max_depth = 0;
	for (int i = 0; i < cloud.size(); ++i) {
		const float& z = cloud[i].z;
		if (z < min_depth) min_depth = z;
		if (z > max_depth) max_depth = z;
	}

	return cloudToColorDepthImage(cloud, min_depth, max_depth, nanColor);
}

template<typename PointT>
cv::Mat normalCloudToImage(const pcl::PointCloud<PointT>& cloud, const Eigen::Vector3f& lightVector, bool do_abs = false, cv::Vec3b nanColor = cv::Vec3b(0,0,0))
{
	unsigned int rows = cloud.height;
	unsigned int cols = cloud.width;

	cv::Mat image_color(rows, cols, CV_8UC3);
	Eigen::Vector3f vecToLight = -lightVector.normalized();

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const PointT& p = cloud.at(col, row);

			if (pcl_isnan(p.normal_z)) {
				image_color.at<cv::Vec3b>(row, col) = nanColor;
			}
			else {
				float ambient = 0.2;
				float diffuse = 1.0 - ambient;
				float dot_prod = p.getNormalVector3fMap().dot(vecToLight);
				float intensity = do_abs ? ambient + diffuse * fabs(dot_prod) : ambient + diffuse * std::max(0.f, dot_prod);
				image_color.at<cv::Vec3b>(row, col)[0] = intensity * 255;
				image_color.at<cv::Vec3b>(row, col)[1] = intensity * 255;
				image_color.at<cv::Vec3b>(row, col)[2] = intensity * 255;
			}
		}
	}

	return image_color;
}

template<typename PointT>
cv::Mat normalCloudToRGBImage(const pcl::PointCloud<PointT>& cloud, cv::Vec3b nanColor = cv::Vec3b(0,0,0))
{
	unsigned int rows = cloud.height;
	unsigned int cols = cloud.width;

	cv::Mat image_color(rows, cols, CV_8UC3);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const PointT& p = cloud.at(col, row);

			if (pcl_isnan(p.normal_z)) {
				image_color.at<cv::Vec3b>(row, col) = nanColor;
			}
			else {
				image_color.at<cv::Vec3b>(row, col)[0] = (p.normal_x + 1.0) / 2 * 255;
				image_color.at<cv::Vec3b>(row, col)[1] = (p.normal_y + 1.0) / 2 * 255;
				image_color.at<cv::Vec3b>(row, col)[2] = (p.normal_z + 1.0) / 2 * 255;
			}
		}
	}

	return image_color;
}

template<typename PointT>
typename pcl::PointCloud<PointT>::Ptr depthImageToCloud(const Parameters& params, const cv::Mat& image, const cv::Rect& rect)
{
	unsigned int rows = image.rows;
	unsigned int cols = image.cols;

	typename pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>);
	result->width = cols;
	result->height = rows;
	result->is_dense = false;
	result->resize(rows * cols);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const float& z = image.at<float>(row, col);
			PointT& p = result->at(col, row);

			if (pcl_isnan(z)) {
				p.x = p.y = p.z = z;
			}
			else {
				int col_image = col;
				int row_image = row;
				if (rect != cv::Rect()) {
					col_image += rect.x;
					row_image += rect.y;
				}

				p.z = z;
				p.x = (col_image - params.camera_center_x) * z / params.camera_focal_x;
				p.y = (row_image - params.camera_center_y) * z / params.camera_focal_y;
			}

			// just white for now
			p.r = p.g = p.b = 255;
		}
	}

	return result;
}

template<typename PointT>
cv::Mat cloudToDepthImage(const pcl::PointCloud<PointT>& cloud)
{
	unsigned int rows = cloud.height;
	unsigned int cols = cloud.width;

	cv::Mat image_depth(rows, cols, CV_32FC1);

	for (unsigned int row = 0; row < rows; row++) {
		for (unsigned int col = 0; col < cols; col++) {
			const PointT& p = cloud.at(col, row);
			image_depth.at<float>(row, col) = p.z;
		}
	}

	return image_depth;
}