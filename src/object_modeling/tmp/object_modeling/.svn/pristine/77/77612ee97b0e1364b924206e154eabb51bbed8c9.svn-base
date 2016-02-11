#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"
#include "VolumeBuffer.h"

class KernelExtractIntForPointImage
{
public:
	static const std::string kernel_name;

	KernelExtractIntForPointImage(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		VolumeBuffer & volume,
		ImageBuffer & points_image,
		ImageBuffer & result_image,
		Eigen::Affine3f const& model_pose,
		float voxel_size
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};