// GenerateTestData.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <EigenUtilities.h>

using std::cout;
using std::endl;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> CloudT;

int cols = 640;
int rows = 480;
float focal = 525.0f;
float center_x = 0.5f * (cols - 1);
float center_y = 0.5f * (rows - 1);

Eigen::Vector3f get3DPoint(int u, int v, float z)
{
	Eigen::Vector3f result;
	result.z() = z;
	result.x() = ((float)u - center_x) * z / focal;
	result.y() = ((float)v - center_y) * z / focal;
	return result;
}

Eigen::Vector2f get2DPoint(Eigen::Vector3f const& p)
{
	Eigen::Vector2f result;
	result[0] = p.x() * focal / p.z() + center_x;
	result[1] = p.y() * focal / p.z() + center_y;
	return result;
}

CloudT::Ptr getEmptyKinectCloud()
{
	CloudT::Ptr result(new CloudT);
	result->width = cols;
	result->height = rows;
	result->resize(result->width * result->height);
	result->is_dense = false;
	// others?

	// Make setting nan easy:
	Eigen::Vector3f nanpoint(std::numeric_limits<float>::quiet_NaN(), 
		std::numeric_limits<float>::quiet_NaN(), 
		std::numeric_limits<float>::quiet_NaN());

	for (CloudT::iterator iter = result->begin(); iter != result->end(); ++iter) {
		iter->getVector3fMap() = nanpoint;
	}

	return result;
}

void generateSet0(const fs::path& output_folder)
{
	std::ofstream ofs_input_pose( (output_folder / "object_poses.txt").string() );
	std::ofstream ofs_loop_pose( (output_folder / "loop_poses.txt").string() );
	std::ofstream ofs_loop_which_pvs( (output_folder / "loop_which_pvs.txt").string() );
	Eigen::Affine3f object_pose_offset = Eigen::Affine3f::Identity();
	Eigen::Vector3f object_center = Eigen::Vector3f(0,0,1);
	object_pose_offset.translate(object_center);
	float edge_length = 0.2; // m
	float l2 = edge_length / 2;

	for (uint32_t i = 0; i < 5; ++i) {
		Eigen::Vector3i color = Eigen::Vector3i(255,255,255);
		if (i == 1) color = Eigen::Vector3i(255,0,255);
		else if (i == 2) color = Eigen::Vector3i(0,255,0);
		else if (i == 3) color = Eigen::Vector3i(0,0,255);

		// five views of a "cube"
		fs::path filepath = output_folder / (boost::format("%05d.pcd") % i).str();
		CloudT::Ptr cloud_ptr = getEmptyKinectCloud();
		
		float l2 = edge_length / 2;
		Eigen::Vector2f ul = get2DPoint(object_center + Eigen::Vector3f(-l2,-l2,-l2));
		Eigen::Vector2f lr = get2DPoint(object_center + Eigen::Vector3f(l2,l2,-l2));

		for (int row = ul[1]; row < lr[1]; ++row) {
			for (int col = ul[0]; col < lr[0]; ++col) {
				PointT& p = cloud_ptr->at(col,row);
				p.getVector3fMap() = get3DPoint(col, row, object_center[2] - l2);
				p.r = color[0];
				p.g = color[1];
				p.b = color[2];
			}
		}
		pcl::io::savePCDFileBinary(filepath.string(), *cloud_ptr);

		// pose...rotate by a slightly wrong amount every time?
		Eigen::Affine3f input_object_pose = object_pose_offset;
		if (i > 0) {
			float wrong_factor = 0.90;
			Eigen::Affine3f r;
			r = Eigen::AngleAxisf( (wrong_factor * M_PI_2) * i, Eigen::Vector3f(0,-1,0));
			input_object_pose = object_pose_offset * r;
		}
		ofs_input_pose << EigenUtilities::transformToString(input_object_pose) << endl;

		// loop closure pose happens on last frame, telling correct pose
		if (i == 4) {
			Eigen::Affine3f loop_pose = object_pose_offset; // no rotation
			ofs_loop_pose << EigenUtilities::transformToString(loop_pose) << endl;
			ofs_loop_which_pvs << "1" << endl; // remember trivial 0 doesn't count..
		}
		else {
			ofs_loop_pose << endl;
			ofs_loop_which_pvs << endl;
		}
	}
}

void generateSet1(const fs::path& output_folder)
{
	std::ofstream ofs_input_pose( (output_folder / "object_poses.txt").string() );
	std::ofstream ofs_loop_pose( (output_folder / "loop_poses.txt").string() );
	std::ofstream ofs_loop_which_pvs( (output_folder / "loop_which_pvs.txt").string() );
	Eigen::Affine3f object_pose_offset = Eigen::Affine3f::Identity();
	Eigen::Vector3f object_center = Eigen::Vector3f(0,0,1);
	Eigen::Affine3f transform_all_poses = Eigen::Affine3f::Identity();
	// as a bug test, try camera not even starting at identity:
	//transform_all_poses = transform_all_poses * Eigen::AngleAxisf(M_PI / 6, Eigen::Vector3f(0,-1,0));
	//transform_all_poses = transform_all_poses * Eigen::Translation3f(0.2,0.2,0.2);
	object_pose_offset.translate(object_center);
	float edge_length = 0.2; // m
	float l2 = edge_length / 2;

	for (uint32_t i = 0; i < 5; ++i) {
		Eigen::Vector3i color = Eigen::Vector3i(255,255,255);
		if (i == 1) color = Eigen::Vector3i(255,0,255);
		else if (i == 2) color = Eigen::Vector3i(0,255,0);
		else if (i == 3) color = Eigen::Vector3i(0,0,255);

		// five views of a "split cube"
		fs::path filepath = output_folder / (boost::format("%05d.pcd") % i).str();
		CloudT::Ptr cloud_ptr = getEmptyKinectCloud();
		
		float l2 = edge_length / 2;
		Eigen::Vector2f ul = get2DPoint(object_center + Eigen::Vector3f(-l2,-l2,-l2));
		Eigen::Vector2f lr = get2DPoint(object_center + Eigen::Vector3f(l2,l2,-l2));

		for (int row = ul[1]; row < lr[1]; ++row) {
			for (int col = ul[0]; col < lr[0]; ++col) {
				// split by skipping a single row and column of pixels in the center?
				if (row == (int)center_y || col == (int)center_x) continue;

				PointT& p = cloud_ptr->at(col,row);
				p.getVector3fMap() = get3DPoint(col, row, object_center[2] - l2);
				p.r = color[0];
				p.g = color[1];
				p.b = color[2];
			}
		}
		pcl::io::savePCDFileBinary(filepath.string(), *cloud_ptr);

		// pose...rotate by a slightly wrong amount every time?
		Eigen::Affine3f input_object_pose = object_pose_offset;
		if (i > 0) {
			float wrong_factor = 0.90;
			Eigen::Affine3f r;
			r = Eigen::AngleAxisf( (wrong_factor * M_PI_2) * i, Eigen::Vector3f(0,-1,0));
			input_object_pose = object_pose_offset * r;
		}
		input_object_pose =  input_object_pose * transform_all_poses;
		ofs_input_pose << EigenUtilities::transformToString(input_object_pose) << endl;

		// loop closure pose happens on last frame, telling correct pose
		if (i == 4) {
			Eigen::Affine3f loop_pose = object_pose_offset; // no rotation
			loop_pose =  loop_pose * transform_all_poses;
			ofs_loop_pose << EigenUtilities::transformToString(loop_pose) << endl;
			ofs_loop_which_pvs << "1 2 3 4" << endl; // remember trivial 0 doesn't count..
		}
		else {
			ofs_loop_pose << endl;
			ofs_loop_which_pvs << endl;
		}
	}
}

int _tmain(int argc, _TCHAR* argv[])
{
	fs::path output_folder = "dump";
	fs::create_directories(output_folder);
	generateSet1(output_folder);
	return 0;
}

