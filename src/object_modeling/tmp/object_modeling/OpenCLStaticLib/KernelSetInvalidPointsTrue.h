#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

class KernelSetInvalidPointsTrue
{
public:
	static const std::string kernel_name;

	KernelSetInvalidPointsTrue(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		const ImageBuffer & depth_image,
		ImageBuffer & bool_image,
		bool resize_bool_image
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
