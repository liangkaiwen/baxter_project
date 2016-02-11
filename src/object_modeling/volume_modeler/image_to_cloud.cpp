#include "image_to_cloud.h"

Eigen::Array2f pointToPixel(const ParamsCamera & params_camera, const Eigen::Vector4f & p)
{
	Eigen::Vector2f result;
	result = (p.head<2>().array() * params_camera.focal / p[2]) + params_camera.center;
	return result;
}

Eigen::Vector4f depthToPoint(const ParamsCamera & params_camera, const Eigen::Array2f & pixel, const float & d)
{
	Eigen::Vector4f p;
	p.head<2>() = (pixel - params_camera.center) * d / params_camera.focal;
	p[2] = d;
	p[3] = 1;
	return p;
}

void imageToCloud(const ParamsCamera & params_camera, const cv::Mat & depth_image, Eigen::Matrix4Xf & cloud)
{
	cloud = Eigen::Matrix4Xf(4, depth_image.total());
	cloud.fill(std::numeric_limits<float>::quiet_NaN());
	for (int row = 0; row < depth_image.rows; ++row) {
		for (int col = 0; col < depth_image.cols; ++col) {
			float d = depth_image.at<float>(row, col);
			if (d > 0) {
				cloud.col(row * depth_image.cols + col) = depthToPoint(params_camera, Eigen::Array2f(col,row), d);
			}
		}
	}
}