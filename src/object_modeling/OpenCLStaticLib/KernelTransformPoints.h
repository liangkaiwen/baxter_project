#pragma once

#include "OpenCLAllKernels.h"

#include "ImageBuffer.h"

class KernelTransformPoints
{
public:
	static const std::string kernel_name;

	KernelTransformPoints(OpenCLAllKernels & opencl_kernels);

	void runKernel(
		const ImageBuffer & input_points,
		ImageBuffer & output_points,
		const Eigen::Affine3f & pose
		);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
