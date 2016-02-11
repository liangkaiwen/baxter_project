#pragma once

#include "OpenCLAllKernels.h"

class KernelPickIfIndexFloat4
{
public:
	static const std::string kernel_name;

    KernelPickIfIndexFloat4(OpenCLAllKernels & opencl_kernels);

	void runKernel(cl::Buffer & running_buffer, cl::Buffer & possible_index, cl::Buffer & possible_values, int index, size_t size);

protected:
	CL & cl_;
	cl::Kernel kernel_;
};
