#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelSetVolumeSDFBox
{
public:
	static const std::string kernel_name;

	KernelSetVolumeSDFBox(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume,
		const float voxel_size,
		const Eigen::Affine3f& volume_pose,
		const Eigen::Array3f& box_corner_from_origin,
		const Eigen::Affine3f& box_pose,
		const Eigen::Array2i& camera_size,
		const Eigen::Array2f& camera_focal,
		const Eigen::Array2f& camera_center
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
