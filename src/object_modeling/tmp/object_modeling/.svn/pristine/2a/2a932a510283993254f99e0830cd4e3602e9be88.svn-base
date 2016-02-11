#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRenderColorForPoints
{
public:
	static const std::string kernel_name;

	KernelRenderColorForPoints(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume_c,
		ImageBuffer & rendered_mask,
		ImageBuffer & rendered_points,
		ImageBuffer & rendered_colors,
		float voxel_size,
		Eigen::Affine3f const& model_pose,
		int mask_value
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
