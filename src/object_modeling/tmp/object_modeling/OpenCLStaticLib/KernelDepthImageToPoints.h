#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelDepthImageToPoints
{
public:
	static const std::string kernel_name;

	KernelDepthImageToPoints(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		const ImageBuffer & depth_image,
		ImageBuffer & points_image,
		Eigen::Array2f const& camera_focal,
		Eigen::Array2f const& camera_center
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
