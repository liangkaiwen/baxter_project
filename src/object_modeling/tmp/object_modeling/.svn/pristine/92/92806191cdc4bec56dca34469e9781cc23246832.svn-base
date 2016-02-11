#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

template <typename PointT>
void changeFocalLength(pcl::PointCloud<PointT>& cloud, float new_focal_x, float new_focal_y, float new_center_x, float new_center_y)
{
	for (int row = 0; row < cloud.height; row++) {
		for (int col = 0; col < cloud.width; col++) {
			PointT& p = cloud.at(col, row);
			if (pcl_isnan(p.z)) continue;
			p.x = p.z / new_focal_x * (col - new_center_x);
			p.y = p.z / new_focal_y * (row - new_center_y);
		}
	}
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr projectRenderCloud(pcl::PointCloud<PointT> const& cloud, Eigen::Vector2f const& f_vec, Eigen::Vector2f const& c_vec, Eigen::Vector2f const& offset_vec)
{
	typename pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>);
	PointT empty_point;
	empty_point.x = empty_point.y = empty_point.z = std::numeric_limits<float>::quiet_NaN();
	result->points.assign(cloud.size(), empty_point);
	result->width = cloud.width;
	result->height = cloud.height;
	result->is_dense = cloud.is_dense;

	// This is terrible code...fix it sometime:
	for (int row = 0; row < cloud.height; ++row) {
		for (int col = 0; col < cloud.width; ++col) {
			PointT const& p = cloud.at(col,row);
			if (pcl_isfinite(p.z)) {
				//Eigen::Vector2f p_before_proj = p.getVector3fMap().head<2>();
				Eigen::Vector2f proj;
				proj.x() = p.x * f_vec.x() / p.z + c_vec.x();
				proj.y() = p.y * f_vec.y() / p.z + c_vec.y();
				proj.x() -= offset_vec.x();
				proj.y() -= offset_vec.y();
				int u = floor(proj.x()+0.5);
				int v = floor(proj.y()+0.5);
				if (u >= 0 && u < result->width && v >= 0 && v < result->height) {
					PointT & r = result->at(u,v);
					if (pcl_isnan(r.z) || p.z < r.z) {
						r = p;
					}
				}
			}
		}
	}

	return result;
}


template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr maskCloud(pcl::PointCloud<PointT> const& cloud, cv::Mat const& mask)
{
	typename pcl::PointCloud<PointT>::Ptr result(new pcl::PointCloud<PointT>);
	PointT empty_point;
	empty_point.x = empty_point.y = empty_point.z = std::numeric_limits<float>::quiet_NaN();

#if 0
	result->points.assign(cloud.size(), empty_point);
	result->width = cloud.width;
	result->height = cloud.height;
	result->is_dense = cloud.is_dense;
#endif
	*result = cloud;

	// This is terrible code...fix it sometime:
	for (int row = 0; row < cloud.height; ++row) {
		for (int col = 0; col < cloud.width; ++col) {
			if (!mask.at<uchar>(row,col)) {
				result->at(col,row) = empty_point;
			}
		}
	}

	return result;
}