#pragma once

#include "OpenCLAllKernels.h"

class KernelSetUChar
{
public:
	static const std::string kernel_name;

	KernelSetUChar(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & buffer, size_t size, unsigned char value);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};