#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelRaytraceBox
{
public:
	static const std::string kernel_name;

	KernelRaytraceBox(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		ImageBuffer & rendered_mask,
		ImageBuffer & rendered_points,
		ImageBuffer & rendered_normals,
		const Eigen::Affine3f & model_pose,
		const Eigen::Array3f& box_corner_from_origin,
		const Eigen::Affine3f& box_pose,
		const Eigen::Array2f & camera_focal,
		const Eigen::Array2f & camera_center,
		float min_render_depth,
		float max_render_depth,
		int mask_value
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
