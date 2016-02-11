#pragma once

#include "OpenCLAllKernels.h"

class KernelSetFloat
{
public:
	static const std::string kernel_name;

	KernelSetFloat(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & buffer, size_t size, float value);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};