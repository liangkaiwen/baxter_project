#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRender2MeansAbs
{
public:
	static const std::string kernel_name;

    KernelRender2MeansAbs(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume_1,
		VolumeBuffer & volume_weights_1,
		VolumeBuffer & volume_2,
		VolumeBuffer & volume_weights_2,
		ImageBuffer & rendered_mask,
		ImageBuffer & rendered_points,
		ImageBuffer & rendered_normals,
		float voxel_size,
		Eigen::Affine3f const& model_pose,
		Eigen::Array2f const& camera_focal,
		Eigen::Array2f const& camera_center,
		float min_render_depth,
		float max_render_depth,
		int mask_value
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
