#pragma once

#include "OpenCLAllKernels.h"

class KernelDivideFloats
{
public:
	static const std::string kernel_name;

	KernelDivideFloats(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & buffer, cl::Buffer & divisors, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};