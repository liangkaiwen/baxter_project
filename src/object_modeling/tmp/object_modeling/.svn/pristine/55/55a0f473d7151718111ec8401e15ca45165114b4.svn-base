#pragma once

#include "OpenCLAllKernels.h"

class KernelMaxFloats
{
public:
	static const std::string kernel_name;

	KernelMaxFloats(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & running_values, cl::Buffer & running_index, cl::Buffer & new_values, int new_index, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};