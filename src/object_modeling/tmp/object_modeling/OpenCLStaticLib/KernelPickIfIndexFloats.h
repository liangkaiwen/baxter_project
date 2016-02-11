#pragma once

#include "OpenCLAllKernels.h"

class KernelPickIfIndexFloats
{
public:
	static const std::string kernel_name;

	KernelPickIfIndexFloats(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & running_buffer, cl::Buffer & possible_index, cl::Buffer & possible_values, int index, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};