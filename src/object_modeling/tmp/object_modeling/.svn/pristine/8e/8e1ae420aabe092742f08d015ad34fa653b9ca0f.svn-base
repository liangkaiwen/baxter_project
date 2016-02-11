#include "stdafx.h"

#include "util.h"

#include <iostream>
using std::cout;
using std::endl;


size_t getVolumeIndex(const Eigen::Array3i & volume_cell_counts, const Eigen::Array3i & p)
{
	return p[2] * volume_cell_counts[0] * volume_cell_counts[1] + p[1] * volume_cell_counts[0] + p[0];
}

bool isVertexInVolume(const Eigen::Array3i & volume_cell_counts, const Eigen::Array3i & p)
{
	return ( (p >= 0).all() && (p < volume_cell_counts).all() );
}

void convertPointsAndFloatsToMeshVertices(std::vector<std::pair<Eigen::Vector3f, float> > const& points_and_d, MeshVertexVector & result)
{
	result.clear();
	result.resize(points_and_d.size());
	for (int i = 0; i < points_and_d.size(); ++i) {
		MeshVertex & v = result[i];
		v.p.head<3>() = points_and_d[i].first;
		v.p[3] = 1;
		v.c[0] = v.c[1] = v.c[2] = 0;
		if (points_and_d[i].second > 0) {
			v.c[1] = 255;
		}
		else {
			v.c[0] = 255;
		}
		v.n = Eigen::Vector4f::Zero();
	}
}

void convertPointsAndFloatsToMeshVertices(std::vector<std::pair<Eigen::Vector3f, float> > const& points_and_d, Eigen::Array4ub const& color, MeshVertexVector & result)
{
    result.clear();
    result.resize(points_and_d.size());
    for (int i = 0; i < points_and_d.size(); ++i) {
        MeshVertex & v = result[i];
        v.p.head<3>() = points_and_d[i].first;
        v.p[3] = 1;
        v.c = color;
        v.n = Eigen::Vector4f::Zero();
    }
}

void getLinesForBoxCorners(std::vector<Eigen::Vector3f> const& corners, Eigen::Vector4ub const& color, MeshVertexVector & vertices)
{
	vertices.clear();
	
	// first put the lines in
	std::vector<Eigen::Vector3f> line_points;

	// near
	line_points.push_back(corners[0]);
	line_points.push_back(corners[1]);
	line_points.push_back(corners[1]);
	line_points.push_back(corners[3]);
	line_points.push_back(corners[3]);
	line_points.push_back(corners[2]);
	line_points.push_back(corners[2]);
	line_points.push_back(corners[0]);

	// far
	line_points.push_back(corners[4]);
	line_points.push_back(corners[5]);
	line_points.push_back(corners[5]);
	line_points.push_back(corners[7]);
	line_points.push_back(corners[7]);
	line_points.push_back(corners[6]);
	line_points.push_back(corners[6]);
	line_points.push_back(corners[4]);

	// sides
	line_points.push_back(corners[0]);
	line_points.push_back(corners[4]);
	line_points.push_back(corners[1]);
	line_points.push_back(corners[5]);
	line_points.push_back(corners[2]);
	line_points.push_back(corners[6]);
	line_points.push_back(corners[3]);
	line_points.push_back(corners[7]);

	// now loop over line points and put them in the vertices
	vertices.resize(line_points.size());
	for (int i = 0; i < vertices.size(); ++i) {
		MeshVertex & v = vertices[i];
		v.p.head<3>() = line_points[i];
		v.p[3] = 1.f;
		v.n = Eigen::Vector4f::Zero();
		v.c = color;
	}
}

void getMeshForBoxCorners(std::vector<Eigen::Vector3f> const& corners, Eigen::Vector4ub const& color, MeshVertexVector & vertices, TriangleVector & triangles)
{
	vertices.clear();
	triangles.clear();

	// vertices are just corners
	vertices.resize(8);
	for (int i = 0; i < 8; ++i) {
		MeshVertex & v = vertices[i];
		v.p.head<3>() = corners[i];
		v.p[3] = 1.f;
		// 0 normal ok?
		v.n = Eigen::Vector4f::Zero();
		// color argument
		v.c = color;
	}

	triangles.resize(12);
	int t = 0;
	// left
	triangles[t++] = Eigen::Array3i(0,1,3);
	triangles[t++] = Eigen::Array3i(0,3,2);
	// right
	triangles[t++] = Eigen::Array3i(4,7,5);
	triangles[t++] = Eigen::Array3i(4,6,7);
	// front
	triangles[t++] = Eigen::Array3i(0,2,6);
	triangles[t++] = Eigen::Array3i(0,6,4);
	// back
	triangles[t++] = Eigen::Array3i(1,5,7);
	triangles[t++] = Eigen::Array3i(1,7,3);
	// top
	triangles[t++] = Eigen::Array3i(0,5,1);
	triangles[t++] = Eigen::Array3i(0,4,5);
	// bottom
	triangles[t++] = Eigen::Array3i(2,3,7);
	triangles[t++] = Eigen::Array3i(2,7,6);
}

void makeContinousLine(const std::vector<Eigen::Vector3f> & points, std::vector<Eigen::Vector3f> & result)
{
    result.clear();
    if (points.empty()) return;
    result.push_back(points[0]);
    for (size_t i = 1; i < points.size(); ++i) {
        result.push_back(points[i]);
        result.push_back(points[i]);
    }
    result.push_back(points[0]);
}

void getSliceLines(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, int axis, int position, const Eigen::Vector4ub & color, MeshVertexVector & vertices)
{
    vertices.clear();

    Eigen::Array3i box_min(0,0,0);
    Eigen::Array3i box_max = (volume_cell_counts - 1);

    std::vector<Eigen::Vector3f> corners;
    if (axis == 0) {
        corners.push_back(Eigen::Vector3f(position, box_min[1], box_min[2]));
        corners.push_back(Eigen::Vector3f(position, box_min[1], box_max[2]));
        corners.push_back(Eigen::Vector3f(position, box_max[1], box_max[2]));
        corners.push_back(Eigen::Vector3f(position, box_max[1], box_min[2]));
    }
    else if (axis == 1) {
        corners.push_back(Eigen::Vector3f(box_min[0], position, box_min[2]));
        corners.push_back(Eigen::Vector3f(box_min[0], position, box_max[2]));
        corners.push_back(Eigen::Vector3f(box_max[0], position, box_max[2]));
        corners.push_back(Eigen::Vector3f(box_max[0], position, box_min[2]));
    }
    else if (axis == 2) {
        corners.push_back(Eigen::Vector3f(box_min[0], box_min[1], position));
        corners.push_back(Eigen::Vector3f(box_min[0], box_max[1], position));
        corners.push_back(Eigen::Vector3f(box_max[0], box_max[1], position));
        corners.push_back(Eigen::Vector3f(box_max[0], box_min[1], position));
    }
    else {
        cout << "bad axis in getSliceLines" << endl;
        return;
    }

    std::vector<Eigen::Vector3f> corner_lines;
    makeContinousLine(corners, corner_lines);

    for (size_t i = 0; i < corner_lines.size(); ++i) {
        MeshVertex v;
        v.p.head<3>() = pose * (corner_lines[i] * volume_cell_size);
        v.p[3] = 1;
        v.c = color;
        v.n = Eigen::Vector4f::Zero();
        vertices.push_back(v);
    }
}

void getBoundingLines(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, const Eigen::Array4ub & color, MeshVertexVector & vertices)
{
	std::vector<Eigen::Vector3f> corners = getVolumeCorners(volume_cell_counts, volume_cell_size, pose);
	getLinesForBoxCorners(corners, color, vertices);
}

void getBoundingMesh(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f & pose, const Eigen::Array4ub & color, MeshVertexVector & vertices, TriangleVector & triangles)
{
	std::vector<Eigen::Vector3f> corners = getVolumeCorners(volume_cell_counts, volume_cell_size, pose);
	getMeshForBoxCorners(corners, color, vertices, triangles);
}

std::vector<Eigen::Vector3f> getVolumeCorners(const Eigen::Array3i & volume_cell_counts, const float & volume_cell_size, const Eigen::Affine3f& pose)
{
	Eigen::Array3f box_min(0,0,0);
	Eigen::Array3f box_max = (volume_cell_counts - 1).cast<float>() * volume_cell_size;
	return getBoxCorners(box_min, box_max, pose);
}

std::vector<Eigen::Vector3f> getBoxCorners(Eigen::Array3f const& box_min, Eigen::Array3f const& box_max, Eigen::Affine3f const& pose)
{
	std::vector<Eigen::Vector3f> result;

	result.push_back(pose * (Eigen::Vector3f(box_min[0], box_min[1], box_min[2])));
	result.push_back(pose * (Eigen::Vector3f(box_min[0], box_min[1], box_max[2])));
	result.push_back(pose * (Eigen::Vector3f(box_min[0], box_max[1], box_min[2])));
	result.push_back(pose * (Eigen::Vector3f(box_min[0], box_max[1], box_max[2])));
	result.push_back(pose * (Eigen::Vector3f(box_max[0], box_min[1], box_min[2])));
	result.push_back(pose * (Eigen::Vector3f(box_max[0], box_min[1], box_max[2])));
	result.push_back(pose * (Eigen::Vector3f(box_max[0], box_max[1], box_min[2])));
	result.push_back(pose * (Eigen::Vector3f(box_max[0], box_max[1], box_max[2])));

	return result;
}

// I'm sure I defined this somewhere else too
cv::Scalar eigenToCVColor(const Eigen::Array4ub & color_eigen)
{
    return cv::Scalar(color_eigen[0], color_eigen[1], color_eigen[2]); // 4th element???
}

void getAABB(Eigen::Array3i const& volume_cell_counts, float const& volume_cell_size, Eigen::Array3f & min_point, Eigen::Array3f & max_point)
{
    min_point = Eigen::Array3f(0,0,0);
    max_point = min_point + (volume_cell_counts-1).cast<float>() * volume_cell_size;
}

void projectColorsAndDepths(const ParamsCamera & params_camera, const cv::Mat & source_points, const cv::Mat & source_colors, cv::Mat & result_depth, cv::Mat & result_colors)
{
	result_depth = cv::Mat(source_points.size(), CV_32F, cv::Scalar::all(0));
	result_colors = cv::Mat(source_points.size(), CV_8UC4, cv::Scalar::all(0));

	for (int row = 0; row < source_points.rows; ++row) {
		for (int col = 0; col < source_points.cols; ++col) {
			const cv::Vec4f & p = source_points.at<cv::Vec4f>(row,col);
			if (p[2] > 0) {
				Eigen::Vector4f p_eigen(p[0],p[1],p[2],p[3]);
				Eigen::Array2f pixel = p_eigen.head<2>().array() * params_camera.focal / p[2] + params_camera.center;
				Eigen::Array2i pixel_round = (pixel + 0.5).cast<int>();
				if ( (pixel_round >= 0).all() && (pixel_round < params_camera.size).all()) {
					float previous_depth = result_depth.at<float>(pixel_round.y(), pixel_round.x());
					if (previous_depth <= 0 || p_eigen.z() < previous_depth) {
						result_colors.at<cv::Vec4b>(pixel_round.y(), pixel_round.x()) = source_colors.at<cv::Vec4b>(row,col);
						result_depth.at<float>(pixel_round.y(), pixel_round.x()) = p_eigen.z();
					}
				}
			}
		}
	}
}

// gives in the TARGET image space the depth and mapping back to source pixels
void projectPixels(const ParamsCamera & params_camera, const cv::Mat & source_points, cv::Mat & result_depth, cv::Mat & result_pixels)
{
	result_depth = cv::Mat(source_points.size(), CV_32F, cv::Scalar::all(0));
	result_pixels = cv::Mat(source_points.size(), CV_32SC2, cv::Scalar::all(-1));

	for (int row = 0; row < source_points.rows; ++row) {
		for (int col = 0; col < source_points.cols; ++col) {
			const cv::Vec4f & p = source_points.at<cv::Vec4f>(row,col);
			if (p[2] > 0) {
				Eigen::Vector4f p_eigen(p[0],p[1],p[2],p[3]);
				Eigen::Array2f pixel = p_eigen.head<2>().array() * params_camera.focal / p[2] + params_camera.center;
				Eigen::Array2i pixel_round = (pixel + 0.5).cast<int>();
				if ( (pixel_round >= 0).all() && (pixel_round < params_camera.size).all()) {
					float previous_depth = result_depth.at<float>(pixel_round.y(), pixel_round.x());
					if (previous_depth <= 0 || p_eigen.z() < previous_depth) {
						result_pixels.at<cv::Vec2i>(pixel_round.y(), pixel_round.x()) = cv::Vec2i(col,row);
						result_depth.at<float>(pixel_round.y(), pixel_round.x()) = p_eigen.z();
					}
				}
			}
		}
	}
}

void getVerticesForPointsAndColors(const cv::Mat & points, const cv::Mat & colors_bgra, MeshVertexVector & vertices)
{
	vertices.clear();
	for (int row = 0; row < points.rows; ++row) {
		for (int col = 0; col < points.cols; ++col) {
			const cv::Vec4f & p = points.at<cv::Vec4f>(row,col);
			if (p[2] > 0) {
				MeshVertex v;							
				v.p[0] = p[0];
				v.p[1] = p[1];
				v.p[2] = p[2];
				v.p[3] = p[3];

				const cv::Vec4b & c = colors_bgra.at<cv::Vec4b>(row,col);
				v.c[0] = c[0];
				v.c[1] = c[1];
				v.c[2] = c[2];
				v.c[3] = c[3];

				vertices.push_back(v);
			}
		}
	}
}
