#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRenderNormalForPoints
{
public:
	static const std::string kernel_name;

	KernelRenderNormalForPoints(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume_d,
		VolumeBuffer & volume_dw,
		ImageBuffer & rendered_mask,
		ImageBuffer & rendered_points,
		ImageBuffer & rendered_normals,
		float voxel_size,
		Eigen::Affine3f const& model_pose,
		int mask_value
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
