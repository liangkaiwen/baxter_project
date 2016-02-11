#pragma once

#include "basic.h"

#include "params_camera.h"

Eigen::Array2f pointToPixel(const ParamsCamera & params_camera, const Eigen::Vector4f & p);
Eigen::Vector4f depthToPoint(const ParamsCamera & params_camera, const Eigen::Array2f & pixel, const float & d);
void imageToCloud(const ParamsCamera & params_camera, const cv::Mat & depth_image, Eigen::Matrix4Xf & cloud);

