#pragma once

#include <EigenUtilities.h>
#include <opencv2/opencv.hpp>

#include "MeshTypes.h"

#include "params_camera.h"


size_t getVolumeIndex(const Eigen::Array3i & volume_cell_counts, const Eigen::Array3i & p);

bool isVertexInVolume(const Eigen::Array3i & volume_cell_counts, const Eigen::Array3i & p);

void convertPointsAndFloatsToMeshVertices(std::vector<std::pair<Eigen::Vector3f, float> > const& points_and_d, MeshVertexVector & result);
void convertPointsAndFloatsToMeshVertices(std::vector<std::pair<Eigen::Vector3f, float> > const& points_and_d, Eigen::Array4ub const& color, MeshVertexVector & result);

void makeContinousLine(const std::vector<Eigen::Vector3f> & points, std::vector<Eigen::Vector3f> & result);

void getSliceLines(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, int axis, int position, const Eigen::Vector4ub & color, MeshVertexVector & vertices);

void getLinesForBoxCorners(std::vector<Eigen::Vector3f> const& corners, Eigen::Vector4ub const& color, MeshVertexVector & vertices);

void getMeshForBoxCorners(std::vector<Eigen::Vector3f> const& corners, Eigen::Vector4ub const& color, MeshVertexVector & vertices, TriangleVector & triangles);

void getBoundingLines(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, const Eigen::Array4ub & color, MeshVertexVector & vertices);

void getBoundingMesh(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, const Eigen::Array4ub & color, MeshVertexVector & vertices, TriangleVector & triangles);

std::vector<Eigen::Vector3f> getVolumeCorners(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f& pose);

std::vector<Eigen::Vector3f> getBoxCorners(Eigen::Array3f const& box_min, Eigen::Array3f const& box_max, Eigen::Affine3f const& pose);

cv::Scalar eigenToCVColor(const Eigen::Array4ub & color_eigen);

void getAABB(Eigen::Array3i const& volume_cell_counts, float const& volume_cell_size, Eigen::Array3f & min_point, Eigen::Array3f & max_point);

void projectColorsAndDepths(const ParamsCamera & params_camera, const cv::Mat & source_points, const cv::Mat & source_colors, cv::Mat & result_depth, cv::Mat & result_colors);

// gives in the TARGET image space the depth and mapping back to source pixels
void projectPixels(const ParamsCamera & params_camera, const cv::Mat & source_points, cv::Mat & result_depth, cv::Mat & result_pixels);

void getVerticesForPointsAndColors(const cv::Mat & points, const cv::Mat & colors_bgra, MeshVertexVector & vertices);

