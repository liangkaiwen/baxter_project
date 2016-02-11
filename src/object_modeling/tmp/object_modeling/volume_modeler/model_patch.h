#pragma once

#include "model_grid.h"

struct MergeEdge {
	int a;
	int b;
	float w;
};
bool operator<(const MergeEdge &a, const MergeEdge &b);

class ModelPatch : public ModelGrid 
{
public:
	ModelPatch(boost::shared_ptr<OpenCLAllKernels> all_kernels, VolumeModelerAllParams const& params, boost::shared_ptr<Alignment> alignment_ptr, RenderBuffers const& render_buffers);
	ModelPatch(ModelPatch const& other);

	ModelPatch* clone();

	virtual void updateModel(
		Frame & frame,
		const Eigen::Affine3f & model_pose);

protected:
	// functions
	void getCovarianceRotation(Eigen::Matrix3Xf const& cloud_in_world, 
		Eigen::Vector3f & result_cloud_mean, 
		Eigen::Affine3f & result_rotate_cloud_to_axes);

	void getRequiredTSDFValues(Eigen::Matrix3Xf const& cloud_in_tsdf, 
		Eigen::Array3i & result_cell_counts,
		Eigen::Vector3f & result_center_offset);

	void getCenteringPose(Eigen::Matrix3Xf const& cloud_in_world, 
		Eigen::Vector3f & result_cloud_mean, 
		Eigen::Affine3f & result_rotate_cloud_to_axes,
		Eigen::Array3i & result_cell_counts,
		Eigen::Vector3f & result_center_offset);

	void getSegmentation(const Frame & frame,
		const Eigen::Affine3f & pose,
		cv::Mat & result_render_segments, 
		cv::Mat & result_consistent_segments,
		cv::Mat & result_segments,
		std::map<int, int> & segment_sizes_map,
		std::map<int, Eigen::Vector3f> segment_normals_map);
};